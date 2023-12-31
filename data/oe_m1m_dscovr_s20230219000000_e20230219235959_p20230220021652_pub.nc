CDF  �   
      time          *   Conventions       ACDD-1.3, Spase v2.2.3     title         /DSCOVR Magnetometer Level 2 One Minute Averages    id        Doe_m1m_dscovr_s20230219000000_e20230219235959_p20230220021652_pub.nc   naming_authority      gov.noaa.swpc      program       DSCOVR     summary       }Interplanetary magnetic field observations collected from magnetometer on DSCOVR satellite - 1-minute average of Level 1 data      keywords      _NumericalData.ObservedRegion.Heliosphere.NearEarth, NumericalData.MeasurementType.MagneticField    keywords_vocabulary       Spase v2.2.2   
references        �; DSCOVR TIME SERIES DATA AVERAGES ALGORITHM THEORETICAL BASIS DOCUMENT, v2.4; GSE TO GSM COORDINATE TRANSFORMATION ALGORITHM THEORETICAL BASIS DOCUMENT v2.1      metadata_link         �http://www.ngdc.noaa.gov/docucomp/page?xml=NOAA/NESDIS/NGDC/STP/Space_Weather/iso/xml/satellite-systems_dscovr.xml&view=getDataView&header=none    license       Spase.Access Rights.Open   institution       NOAA   source        DSCOVR Magnetometer Level 1    platform      'Deep Space Climate Observatory (DSCOVR)    
instrument        +boom-mounted triaxial fluxgate magnetometer    history       ,DSCOVR real-time telemetry processing system   	algorithm         FDSCOVR MAGNETOMETER LEVEL 1B DATA ALGORITHM THEORETICAL BASIS DOCUMENT     algorithmVersion      B      algorithmDate         
2015-10-15     processing_level      Level 2    processing_level_description      11-minute average using Hodges-Lehmann M-estimator      date_created      2023-02-20T02:16:52.092Z   date_calibration_data_updated         2023-01-28T00:00:00.000Z   time_coverage_duration        P01D   time_coverage_start       2023-02-19T00:00:00.000Z   time_coverage_end         2023-02-19T23:59:59.000Z   time_coverage_resolution      PT1M   creator_name      Doug Biesecker     creator_type      person     creator_institution       DOC/NOAA/NWS/NCEP/SWPC     creator_email         doug.biesecker@noaa.gov    creator_url       http://www.swpc.noaa.gov/      publisher_name         National Geophysical Data Center   publisher_type        institution    publisher_institution         DOC/NOAA/NESDIS/NGDC   publisher_email       william.rowland@noaa.gov   publisher_url          http://www.ngdc.noaa.gov/dscovr/   records_maximum         �   records_present         �   records_data        �   records_fill             records_missing                    time                description       "date and time for each observation     
short_name        time   C_format      %.13g      units         'milliseconds since 1970-01-01T00:00:00Z    lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   standard_name         time   calendar      	gregorian           7   sample_count                description       /number of full resolution measurements averaged    
short_name        sample_count   C_format      %d     units         samples    lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   	valid_min                	valid_max           �        7   measurement_mode                description       7measurement range selection mode (0 = auto, 1 = manual)    
short_name        mode   C_format      %1d    lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   	valid_min                	valid_max                    7   measurement_range                   description       5measurement range (~4x sensitivity increase per step)      
short_name        range      C_format      %d     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   	valid_min                	valid_max                    7   bt               	   description       )Interplanetary Magnetic Field strength Bt      
short_name        bt     C_format      %.4f   units         nT     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   missing_value         ��i�       	long_name         )bt_interplanetary_magnetic_field_strength      	valid_min                	valid_max                    7    bx_gse               
   description       \Interplanetary Magnetic Field strength Bx component in Geocentric Solar Ecliptic coordinates   
short_name        bx_gse     C_format      %.4f   units         nT     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   missing_value         ��i�       	long_name         -bx_interplanetary_magnetic_field_strength_gse      	valid_min         ��     	valid_max               _CoordinateSystems        GSpase.NumericalData.Parameter.CoordinateSystem.CoordinateSystemName.GSE         7$   by_gse               
   description       \Interplanetary Magnetic Field strength By component in Geocentric Solar Ecliptic coordinates   
short_name        by_gse     C_format      %.4f   units         nT     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   missing_value         ��i�       	long_name         -by_interplanetary_magnetic_field_strength_gse      	valid_min         ��     	valid_max               _CoordinateSystems        GSpase.NumericalData.Parameter.CoordinateSystem.CoordinateSystemName.GSE         7(   bz_gse               
   description       \Interplanetary Magnetic Field strength Bz component in Geocentric Solar Ecliptic coordinates   
short_name        bz_gse     C_format      %.4f   units         nT     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   missing_value         ��i�       	long_name         -bz_interplanetary_magnetic_field_strength_gse      	valid_min         ��     	valid_max               _CoordinateSystems        GSpase.NumericalData.Parameter.CoordinateSystem.CoordinateSystemName.GSE         7,   	theta_gse                	   description       RInterplanetary Magnetic Field clock angle in Geocentric Solar Ecliptic coordinates     
short_name        	theta_gse      C_format      %.4f   units         degrees    lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   missing_value         ��i�       	valid_min         ����   	valid_max            Z   _CoordinateSystems        GSpase.NumericalData.Parameter.CoordinateSystem.CoordinateSystemName.GSE         70   phi_gse              	   description       RInterplanetary Magnetic Field polar angle in Geocentric Solar Ecliptic coordinates     
short_name        phi_gse    C_format      %.4f   units         degrees    lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   missing_value         ��i�       	valid_min                	valid_max           h   _CoordinateSystems        GSpase.NumericalData.Parameter.CoordinateSystem.CoordinateSystemName.GSE         74   bx_gsm               
   description       bInterplanetary Magnetic Field strength Bx component in Geocentric Solar Magnetospheric coordinates     
short_name        bx_gsm     C_format      %.4f   units         nT     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   missing_value         ��i�       	long_name         -bx_interplanetary_magnetic_field_strength_gsm      	valid_min         ��     	valid_max               _CoordinateSystems        GSpase.NumericalData.Parameter.CoordinateSystem.CoordinateSystemName.GSM         78   by_gsm               
   description       bInterplanetary Magnetic Field strength By component in Geocentric Solar Magnetospheric coordinates     
short_name        by_gsm     C_format      %.4f   units         nT     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   missing_value         ��i�       	long_name         -by_interplanetary_magnetic_field_strength_gsm      	valid_min         ��     	valid_max               _CoordinateSystems        GSpase.NumericalData.Parameter.CoordinateSystem.CoordinateSystemName.GSM         7<   bz_gsm               
   description       bInterplanetary Magnetic Field strength Bz component in Geocentric Solar Magnetospheric coordinates     
short_name        bz_gsm     C_format      %.4f   units         nT     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   missing_value         ��i�       	long_name         -bz_interplanetary_magnetic_field_strength_gsm      	valid_min         ��     	valid_max               _CoordinateSystems        GSpase.NumericalData.Parameter.CoordinateSystem.CoordinateSystemName.GSM         7@   	theta_gsm                	   description       XInterplanetary Magnetic Field clock angle in Geocentric Solar Magnetospheric coordinates   
short_name        	theta_gsm      C_format      %.4f   units         degrees    lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   missing_value         ��i�       	valid_min         ����   	valid_max            Z   _CoordinateSystems        GSpase.NumericalData.Parameter.CoordinateSystem.CoordinateSystemName.GSM         7D   phi_gsm              	   description       XInterplanetary Magnetic Field polar angle in Geocentric Solar Magnetospheric coordinates   
short_name        phi_gsm    C_format      %.4f   units         degrees    lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   missing_value         ��i�       	valid_min                	valid_max           h   _CoordinateSystems        GSpase.NumericalData.Parameter.CoordinateSystem.CoordinateSystemName.GSM         7H   backfill_flag                   description       �One or more measurements were backfilled from the spacecraft recorder and therefore were not available to forecasters in real-time     
short_name        backfill_flag      C_format      %d     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   	long_name         backfilled_data_flag   	valid_min                	valid_max                    7L   future_packet_time_flag                 description       rOne or more measurements were extracted from a packet whose timestamp was in the future at the point of processing     
short_name        future_packet_time_flag    C_format      %d     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   	long_name         packet_time_in_future_flag     	valid_min                	valid_max                    7P   old_packet_time_flag                description       }One or more measurements were extracted from a packet whose timestamp was older than the threshold at the point of processing      
short_name        old_packet_time_flag   C_format      %d     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   	long_name         %packet_time_older_than_threshold_flag      	valid_min                	valid_max                    7T   	fill_flag                   description       Fill   
short_name        	fill_flag      C_format      %d     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   	long_name         	fill_flag      	valid_min                	valid_max                    7X   possible_saturation_flag                description       �Possible magnetometer saturation based on a measurement range smaller than the next packet's range or by the mag being in manual range mode.   
short_name        possible_saturation_flag   C_format      %d     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   	long_name         %possible_magnetometer_saturation_flag      	valid_min                	valid_max                    7\   calibration_mode_flag                   description       Instrument in calibration mode     
short_name        calibration_mode_flag      C_format      %d     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   	long_name         calibration_mode_flag      	valid_min                	valid_max                    7`   maneuver_flag                   description       4AOCS non-science mode (spacecraft maneuver/safehold)   
short_name        maneuver_flag      C_format      %d     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   	long_name         /AOCS_non_science_mode_maneuver_or_safehold_flag    	valid_min                	valid_max                    7d   low_sample_count_flag                   description       $Average sample count below threshold   
short_name        low_sample_count_flag      C_format      %d     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   	long_name         )average_sample_count_below_threshold_flag      	valid_min                	valid_max                    7h   overall_quality                 description       ;Overall sample quality (0 = normal, 1 = suspect, 2 = error)    
short_name        overall_quality    C_format      %d     units         n/a    lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale        7lBxfo�@  "          @��=�G�@5��Z=q�I33B��==�G�@��Ϳ�ff��p�B���                                    Bxfo��  
�          @�=q?��?����l���k�BZ�R?��@Z�H�33� ��B���                                    Bxfo��  
�          @�=q?Tz�@=q�`  �W�B��
?Tz�@u�������B�z�                                    Bxfo�2  �          @��?:�H@(���S�
�I�B�
=?:�H@|(��Ǯ��{B���                                    Bxfo��  �          @���?xQ�@   �S�
�L{B���?xQ�@tz�У����
B��                                    Bxfo�~  
R          @�(�?�  @B�\�#33���B��3?�  @z=q�0���Q�B�W
                                    Bxfo�$  *          @���?@  @3�
�H���<�
B��?@  @�  ��=q���\B���                                    Bxfo��  
�          @��>��@>�R�E��6�B�z�>��@����Q���=qB��q                                    Bxfo�p  ~          @�{?��@:=q�<(��2��B��\?��@\)�����tz�B�Ǯ                                    Bxfp
  
�          @�Q�?L��@&ff�N�R�Gz�B��H?L��@w
=��G���p�B���                                    Bxfp�  
�          @�{��\)@,���=p��>(�B�Ǯ��\)@s�
��p����RB�W
                                    Bxfp'b  
�          @��\��R@:=q�/\)�*p�B�33��R@x�ÿn{�T��B�=q                                    Bxfp6  "          @���aG�@>�R�(Q�� �
B���aG�@x�ÿL���5p�B͏\                                    BxfpD�  �          @��R��ff@`�׿��\��Q�B����ff@k�?#�
A=qB�R                                    BxfpST  �          @�33���H@q�?�@�z�B�����H@@��@B�B�                                    Bxfpa�  T          @�zῳ33@q녿�p���ffBܳ3��33@�z�>�p�@��B�aH                                    Bxfpp�  
�          @�{���@p  �33�ܸRB����@���=��
?���B��
                                    BxfpF  	�          @�ff���R@qG��   ��Q�B�uÿ��R@���>�?�z�B�p�                                    Bxfp��  �          @��R�k�@g
=�(���B�uÿk�@�33������  B���                                    Bxfp��  "          @�p���@fff�{��B�{��@�33��Q���
=B�                                      Bxfp�8  
(          @��Ϳ�@S33�,����\B����@��R�5��B�B�                                    Bxfp��  
�          @�
=�{@L(����H���
B����{@g
==�Q�?���B�                                    BxfpȄ  
�          @�
=�!�@O\)�����HC ��!�@h��>��?�(�B���                                    Bxfp�*  �          @��(�@fff�@  �Q�B�8R�(�@`  ?���Am��B�                                    Bxfp��  	`          @���У�@_\)������B�{�У�@z�H>.{@\)B�(�                                    Bxfp�v  �          @��H��p�@N�R�6ff�$�B����p�@�
=�aG��;�B�\                                    Bxfq  �          @��H�k�@P���6ff�$�B�33�k�@���^�R�8��B�L�                                    Bxfq�  
Z          @��\�\)@A��E��5�B�(��\)@����
=�|��B�                                    Bxfq h  �          @�=q?#�
@#33�Y���Q33B�\?#�
@y���ٙ���{B��                                     Bxfq/  T          @���?�G�@33�R�\�Hz�Bc(�?�G�@hQ��p����B��H                                    Bxfq=�  �          @�=q?�G�@   �P���E=qB|?�G�@r�\��������B�\                                    BxfqLZ  	�          @�=q�.{@L���8���(33B��þ.{@��R�p���HQ�B��=                                    Bxfq[   �          @�G�>��@8Q��H���=�B��q>��@�����������B��R                                    Bxfqi�  �          @��H?Tz�@%�Vff�K�
B���?Tz�@z=q�������B��                                    BxfqxL  "          @�33?(��@#33�]p��R��B�  ?(��@{���G����\B��                                    Bxfq��  
�          @��
?
=q@*�H�Z=q�M�B�.?
=q@�Q��33��  B���                                    Bxfq��  
Z          @�(�?�R@%��^{�RffB�
=?�R@|�Ϳ�  ��
=B�8R                                    Bxfq�>  "          @�(�>�@!��c33�XffB�Q�>�@|�Ϳ�����\)B��                                    Bxfq��  
�          @���?(�@6ff�Q��A\)B��
?(�@�33�������\B�
=                                    Bxfq��  �          @�z�?
=@.�R�XQ��I�B��q?
=@�G��������B�z�                                    Bxfq�0  "          @���>���@/\)�[��L�B�u�>���@�=q�����\)B��=                                    Bxfq��  "          @�z�>��@!��c33�X\)B��H>��@|�Ϳ������
B�aH                                    Bxfq�|  �          @��>��@'
=�a��UQ�B��{>��@�Q�����(�B��=                                    Bxfq�"  T          @�p�?��@�R�e��Z�\B���?��@z�H��33�ͅB��f                                    Bxfr
�  
�          @�p�>W
=@-p��Y���M33B�>W
=@��ÿУ����B���                                    Bxfrn  "          @�z�?&ff@%��^{�Q��B��?&ff@|�Ϳ�G�����B�G�                                    Bxfr(  �          @�
=?�R@,���`  �N�RB�(�?�R@�=q��p����\B��3                                    Bxfr6�  
�          @�{?��?���W��w��B��?��@)���Q��{By�                                    BxfrE`  �          @�p�@
=q�
=q�s�
�n�RC���@
=q?���e�Yp�B33                                    BxfrT  
0          @�@�þ�
=�k��b\)C��@��?�Q��Z�H�K  A��                                    Bxfrb�  
�          @��R@!녿W
=�aG��S\)C��H@!�?}p��^�R�O�HA�33                                    BxfrqR  "          @�\)@��aG��fff�m\)C��@�?����P  �K�RB��                                    Bxfr�  T          @��\?J=q?�
=�j�H�q��B���?J=q@^{������B��                                    Bxfr��  
�          @�\)?!G�@2�\�X���GQ�B�u�?!G�@��H��������B�k�                                    Bxfr�D  
�          @�z�?��H@\)�n{�U\)B�
?��H@~�R����{B�{                                    Bxfr��  �          @���?��@   �l���R�Bz�
?��@~�R� ������B�#�                                    Bxfr��            @��
?Y��@p��}p��l�B��3?Y��@w
=�Q�����B�
=                                    Bxfr�6  �          @��?��@{�xQ��f�BQ�?��@u���
��
=B�8R                                    Bxfr��  "          @��H?p��@���z=q�j
=B��=?p��@tz�����B�B�                                    Bxfr�  "          @�=q?�z�@�R�hQ��RQ�BU(�?�z�@mp����p�B���                                    Bxfr�(  
�          @�  ?���?޸R�tz��iG�B=?���@XQ��\)���B�k�                                    Bxfs�  T          @�Q�?fff?����|(��u�
B���?fff@g��   �Q�B�33                                    Bxfst  T          @���?��@G��x���l�
Br?��@j=q��H��B��                                    Bxfs!  �          @�G�?!G�?����H(�B��H?!G�@e�+��\)B���                                    Bxfs/�  �          @��?(�?���Q�p�By��?(�@E�I���4
=B��\                                    Bxfs>f  "          @��R?�z�?�\�{��v�RBb��?�z�@\���%��(�B��q                                    BxfsM  �          @�Q�>�33?����(�8RB�  >�33@dz��/\)��B��
                                    Bxfs[�  
�          @��?333@"�\�c�
�V{B��?333@|(�����ə�B��)                                    BxfsjX  �          @�G�=u@O\)�HQ��0G�B�=u@�33��
=�r{B�W
                                    Bxfsx�  
�          @��\��@6ff�a��J(�Bʳ3��@��R���H���
BÀ                                     Bxfs��  �          @��H�\(�@(��{��l=qB�Ǯ�\(�@tz��Q���ffB�W
                                    Bxfs�J  T          @�33�k�?�����G��w�
B�\�k�@j=q�&ff�
G�B�8R                                    Bxfs��  �          @��\?z�@��xQ��h��B��?z�@z=q�G�����B�{                                    Bxfs��  �          @��
�B�\?�  ����B�B�\@Z�H�E�(\)B�ff                                    Bxfs�<  �          @�G��B�\?�G���33k�B�.�B�\@L���L���4  B���                                    Bxfs��  �          @�33>#�
?�(���p���B��3>#�
@L���R�\�6B��                                    Bxfs߈  
�          @��H��=q?�{��\Bϙ���=q@Fff�Vff�<�B�
=                                    Bxfs�.  �          @�=q��\)?s33��ff�fBԣ׾�\)@>{�\(��D33B��3                                    Bxfs��  T          @�=q?�@��x���k�Bp
=?�@i���(���
B�u�                                    Bxftz  T          @��?Q�?�=q���\�  B���?Q�@dz��,(��\)B���                                    Bxft   T          @�G�?Y��?�z������zG�B���?Y��@fff�'
=���B�z�                                    Bxft(�  �          @���?p��?������|��Bzff?p��@a��+���B�33                                    Bxft7l  "          @�=q?�?�=q���
=B�Q�?�@[��=p��"�B��\                                    BxftF  �          @��H>��?J=q��  �B�L�>��@6ff�dz��M=qB��                                    BxftT�  
�          @��>\?�=q��
=ǮB�>\@Z=q�<���"�\B�(�                                    Bxftc^  
�          @�=q>�\)?�=q���  B���>�\)@fff�1��G�B��                                    Bxftr  T          @��\>�@ff����zG�B�{>�@r�\�#�
��B�(�                                    Bxft��  �          @��?   @�\�����z{B��{?   @n{�$z��	��B�                                    Bxft�P  �          @�G�?p��@  �s�
�eG�B���?p��@s33�G����HB�(�                                    Bxft��  �          @��H?���?��R�|(��m�BkG�?���@hQ�� ���33B�B�                                    Bxft��  �          @��?�(�@Mp��@  �$�RB�� ?�(�@�Q쿎{�aG�B�{                                    Bxft�B  �          @���?xQ�@U�;�� p�B���?xQ�@��H�}p��H  B��q                                    Bxft��  �          @���?k�@U�:�H� �RB�
=?k�@��\�z�H�HQ�B��H                                    Bxft؎  T          @��?5@X���6ff��B�G�?5@��H�c�
�6�HB�\)                                    Bxft�4  �          @�  ?xQ�@]p��,�����B��q?xQ�@�=q�=p���HB���                                    Bxft��  T          @�  ?��@R�\�6ff���B���?��@�  �p���A�B��)                                    Bxfu�  "          @��R?�Q�@W��*=q��B�B�?�Q�@�\)�=p��p�B��3                                    Bxfu&  T          @��R?���@[��(Q���
B�k�?���@��׿0���\)B��                                    Bxfu!�  
�          @���?�Q�@Q��0  ���B�p�?�Q�@�{�^�R�0��B��                                    Bxfu0r  �          @��?���@U�+���RB��f?���@��R�G�� z�B�ff                                    Bxfu?  �          @�Q�>�
=@P  �Dz��,Q�B�(�>�
=@�=q��z��pQ�B���                                    BxfuM�  
�          @�Q�?c�
@P  �@  �'  B�B�?c�
@��ÿ�{�dz�B�u�                                    Bxfu\d  T          @���?0��@QG��A��(B�{?0��@�녿����g33B�                                    Bxfuk
  "          @���?=p�@H���I���1��B���?=p�@�  ��ff��(�B�ff                                    Bxfuy�  T          @���?\)@Mp��E�-�B�L�?\)@�G����H�yG�B�.                                    Bxfu�V  �          @���?=p�@@  �R�\�;=qB�33?=p�@�ff���R��(�B��                                    Bxfu��  "          @���?0��@L���HQ��.��B���?0��@�G���G����HB��R                                    Bxfu��  �          @���>�(�@N�R�H���/�\B��3>�(�@��\��  ��{B��=                                    Bxfu�H  
�          @��<�@<���XQ��Cp�B���<�@�ff��������B�(�                                    Bxfu��  �          @��=�\)@AG��U��?(�B�W
=�\)@�
=�\��Q�B��                                    Bxfuє  
�          @���>u@8Q��`  �I��B�aH>u@�{�޸R��=qB�aH                                    Bxfu�:  
�          @�=q>���@8���`���I33B�
=>���@�ff��  ��=qB���                                    Bxfu��  
�          @�Q�>��@P���G
=�.�B���>��@��H��(��{�B��                                    Bxfu��  "          @�\)<��
@E��0���'�B�\)<��
@�Q쿀  �_�B��                                    Bxfv,  �          @����fff@���n�R�^
=B�#׿fff@w������
B�=q                                    Bxfv�  
�          @����@qG��ff���B��;�@�{���
���
B�=q                                    Bxfv)x  �          @���    @\(��9��� z�B��    @��Ϳu�D��B�{                                    Bxfv8  T          @�녾�Q�@(��tz��dffB�녾�Q�@|(��\)���B��=                                    BxfvF�  "          @�녽�G�@��x���j�B�\)��G�@z=q���
=B�B�                                    BxfvUj  T          @�녾���@H���E��1�B�
=����@��R��G���33B���                                    Bxfvd  T          @��׿=p�@��ÿ�=q����B��)�=p�@���>�@�
=Bǔ{                                    Bxfvr�  "          @��ÿ\(�@~{�����̏\B�uÿ\(�@�p�=�?ǮB�
=                                    Bxfv�\  �          @�=q��\)@vff��ff��
=B����\)@�  >B�\@
=B�                                    Bxfv�  �          @�=q�Ǯ@��\��G�����Bݽq�Ǯ@�ff?E�A��Bܣ�                                    Bxfv��  T          @���Q�@l�Ϳ��
���B�Ǯ�Q�@\)>�Q�@��B�(�                                    Bxfv�N  "          @�33���@��H�W
=�1B�p����@���?�33Ar�HB�\                                    Bxfv��  
�          @����{@C33�P���4G�B�  ��{@��R��p���Bъ=                                    Bxfvʚ  )          @�33�Y��@'
=�k��T�B��f�Y��@����33�ԣ�B���                                    Bxfv�@  "          @���J=q@#�
�n�R�Y33B�.�J=q@�  �Q��ܸRB�Q�                                    Bxfv��  
�          @�33�xQ�@*�H�g��O��B�
=�xQ�@��ÿ�p���Q�B�\                                    Bxfv��  
�          @���p��@\)�p���Z�
Bݙ��p��@|(��(���G�B���                                    Bxfw2  
�          @��H��ff@�\�|(��o33B�8R��ff@hQ��#�
��\B�(�                                    Bxfw�  
�          @�(����\@)���j=q�P��B�8R���\@��������
B�u�                                    Bxfw"~  
�          @�녿G�@N{�Fff�,�\B�k��G�@��ÿ��
��
=B�                                    Bxfw1$  	�          @�녿&ff@<(��Z=q�A��B�녿&ff@���
=��
=Bų3                                    Bxfw?�  �          @��׾���@0���b�\�O�BĮ����@�=q����ŅB�W
                                    BxfwNp  
�          @������H?�p�����}p�B����H@g��-p��(�B�aH                                    Bxfw]  "          @�\)�\)?��
����8RB�=q�\)@P���?\)�(�Bǀ                                     Bxfwk�  �          @�{�
=q?У���=qQ�Bؙ��
=q@S�
�8Q��"��B�z�                                    Bxfwzb  
�          @�=q��ff@���u�dG�B����ff@{��33��B�
=                                    Bxfw�  �          @�녿(��@���tz��cz�B��
�(��@xQ��33��RB�aH                                    Bxfw��  
�          @������@
=q�{��q�B����@n{� ���  B�W
                                    Bxfw�T  
a          @�33�Y��@z��xQ��e��B�(��Y��@u������\)B���                                    Bxfw��  T          @�=q�J=q?�p���G��x�B�uÿJ=q@fff�,�����B̽q                                    Bxfwà  
Z          @�33�xQ�@p��x���h��B�ff�xQ�@n�R�{�ffB�{                                    Bxfw�F  
�          @�33�k�@   �p  �ZffB�=q�k�@{��p���\)B�B�                                    Bxfw��  �          @��H�Y��@(��r�\�_  B�=q�Y��@x���G����
B̅                                    Bxfw�  
�          @��H����@*=q�dz��L\)B�
=����@~�R��p���ffB��                                    Bxfw�8  "          @��H�#�
?�z���\)8RB�G��#�
@Y���AG��$z�B�u�                                    Bxfx�  
�          @���G�?�Q���G�=qB왚�G�@N�R�K��/(�B�{                                    Bxfx�  
�          @��\���?�ff��Q�L�C �R���@E��N{�2�RB�W
                                    Bxfx**  "          @����ff?�{����ffCs3��ff@:�H�Tz��8=qB�                                      Bxfx8�  �          @��ÿ�?��R����3C
Ǯ��@=p��G
=�-�HB�8R                                    BxfxGv  
�          @�z΅{@XQ�����
B��
��{@���&ff���Bؙ�                                    BxfxV  
�          @������@hQ��p���p�Bր ����@�\)��33��  B���                                    Bxfxd�  "          @���B�\@\���-p���B�33�B�\@����^�R�6�\B���                                    Bxfxsh  
�          @�z�>#�
@\(��(����\B���>#�
@�Q�O\)�,��B���                                    Bxfx�  
�          @�ff?G�@fff�p����B��=?G�@�=q�z���=qB��                                    Bxfx��  
�          @�\)>��H@mp��p��
=B��H>��H@�p��
=q����B�G�                                    Bxfx�Z  T          @��R?8Q�@^{�+��33B���?8Q�@����Y���/\)B�{                                    Bxfx�   �          @���?J=q@n�R��噚B�
=?J=q@��׾L���,��B�                                      Bxfx��  "          @���?@  @e��33B�p�?@  @�  �   �љ�B�
=                                    Bxfx�L  �          @��\?J=q@xQ쿷
=����B��?J=q@��>�ff@\B�33                                    Bxfx��  
�          @�(�?L��@g���\��\)B�  ?L��@�Q��G�����B���                                    Bxfx�  �          @�
=?333@8Q��N�R�=��B��
?333@��׿�{���\B�Q�                                    Bxfx�>  T          @�z�?Q�@&ff�p  �X
=B�?Q�@\)�p���B���                                    Bxfy�  
�          @�?�R@R�\�N{�/�B�Ǯ?�R@�(���
=��p�B���                                    Bxfy�  �          @��\��@��H�B�\�=qB��R��@�\)?��RA���B�
=                                    Bxfy#0  �          @�G�=���@��H�!G���B�\)=���@�p�?���A��
B�B�                                    Bxfy1�  �          @�ff�O\)@��\>Ǯ@�G�B�G��O\)@n{@��A�B̸R                                    Bxfy@|  
�          @�p��.{@��\�\)���B��f�.{@�(�?�33A�{BƸR                                    BxfyO"  �          @�zᾊ=q@��ÿY���4  B�=q��=q@�
=?�\)Al��B�W
                                    Bxfy]�  	�          @��Ϳc�
@��þ�G�����B�aH�c�
@�G�?��RA�  B̽q                                    Bxfyln  "          @��
���@��R��\���Bϳ3���@�Q�?��A���B�\                                    Bxfy{  �          @�33�!G�@��׾���=qBĨ��!G�@�G�?��HA�G�Bř�                                    Bxfy��  
(          @����33@��ÿ
=���HB�uþ�33@�33?��A�Q�B��)                                    Bxfy�`  "          @��Ϳ.{@��\���R���B��ÿ.{@���?�{A�ffB�G�                                    Bxfy�  T          @��Ϳ.{@��\�\��{B��)�.{@��?�ffA��B�                                    Bxfy��  
�          @�(��   @�=q��(���{B�=q�   @��\?�  A�{B�                                    Bxfy�R  �          @�z�z�@��H���
���B�(��z�@�G�?���A��B�B�                                    Bxfy��  �          @�(���@��\��p���G�B�
=��@��?ǮA�z�B���                                    Bxfy�  
�          @��
�   @�녾Ǯ����B�W
�   @���?\A�p�B�.                                    Bxfy�D  �          @�33�5@���>aG�@<(�B��)�5@p��@�
A�Q�B�k�                                    Bxfy��  
�          @�=q�\(�@��R>��@�B�#׿\(�@g�@(�A�Q�B���                                    Bxfz�  �          @�z�
=@��H�u�L��B�k��
=@|(�?���A���B���                                    Bxfz6  T          @���
=@�33��{���RB�uÿ
=@�=q?���A�\)BĀ                                     Bxfz*�  T          @��;L��@��
�Ǯ��z�B���L��@��?��
A�G�B�p�                                    Bxfz9�  T          @�(��8Q�@�=q��  �P  B��)�8Q�@�  ?��A��B�Q�                                    BxfzH(  �          @�33�(�@�G��Ǯ���B���(�@�G�?�  A��RB�{                                    BxfzV�  T          @��H��R@�G���  �P  Bď\��R@~{?�\)A���B��
                                    Bxfzet  "          @�=q�!G�@�Q쾣�
���RB��ÿ!G�@~�R?��A��B��                                    Bxfzt  �          @��\�   @�G����R���B�.�   @�Q�?ǮA�z�B��                                    Bxfz��  �          @�33��Q�@��\��z��uB�ff��Q�@���?˅A�p�B�{                                    Bxfz�f  T          @�33�aG�@��H�����B���aG�@~{?�  A�{B�33                                    Bxfz�  
�          @�(���=q@������W
=B����=q@���?У�A���B���                                    Bxfz��  �          @�=q���@������
����B����@z�H?�G�A�=qB���                                    Bxfz�X  
(          @��R��
=@�����{B�z��
=@r�\?�G�A��B���                                    Bxfz��  �          @�G���(�@�
=�
=q��
=B�����(�@���?��A��\B�#�                                    Bxfzڤ  �          @�  �s33@~�R�����i�B�녿s33@���?8Q�AB�k�                                    Bxfz�J  
�          @�p����@l�Ϳ˅��G�B�p����@���>�?���B�
=                                    Bxfz��  T          @����
=@U�
=��  B��H��
=@z�H���H��(�BՀ                                     Bxf{�  "          @��H����@vff����=qB��쿐��@�{=�?˅B�G�                                    Bxf{<  
�          @�\)���H@hQ���
��
=Bؽq���H@�G���\)��G�B�8R                                    Bxf{#�  �          @�\)���@z��QG��KB�W
���@^�R���R��G�B�Ǯ                                    Bxf{2�  
Z          @�33��G�@��A��?�HB�8R��G�@U�����=qB�R                                    Bxf{A.  �          @�����@
=q�,(��(��C@ ��@E���
���B��                                    Bxf{O�  �          @��H��Q�?�{�Z=q�c{C.��Q�@(��(Q�� �\C s3                                    Bxf{^z  �          @�G����H?W
=�Z=q�g�HC�f���H@���/\)�+�
C�3                                    Bxf{m   �          @��
���H?��Z�H�u=qC33���H@   �'
=�(z�B�{                                    Bxf{{�  "          @�녾�  @C33�AG��2�B�.��  @�G���(���B��                                    Bxf{�l  "          @���fff?�z��Z�H�i
=B�uÿfff@J=q��
��B�Ǯ                                    Bxf{�  "          @���z�>B�\�s33�C-����z�?�  �W��Y�Cs3                                    Bxf{��  
Z          @�G����H?�Q��o\)�sB�#׿��H@E�,���=qBޮ                                    Bxf{�^  �          @�33���?��q��s{B�G����@N{�,(��33Bي=                                    Bxf{�  �          @�Q�^�R@�\�e�i�B���^�R@U��H�=qB�
=                                    Bxf{Ӫ  T          @�G��+�@C33�:�H�,ffB̳3�+�@\)��33��=qB�                                    Bxf{�P  �          @��;�G�@[��-p��\)B¨���G�@�  ��ff�\��B��)                                    Bxf{��  �          @����\@k�����Bó3��\@��H�#�
���B�\)                                    Bxf{��  �          @��\�E�@u��=q��{Bʮ�E�@�Q콣�
���Bȅ                                    Bxf|B  
�          @�=q��{@tz��Q����\B�ff��{@�<�>���Bѽq                                    Bxf|�  �          @�  ��33@s�
��  ���HBծ��33@��H>L��@/\)BӀ                                     Bxf|+�  
�          @�\)�k�@xQ쿳33��33BθR�k�@��>��R@�  B�=q                                    Bxf|:4  T          @�=q�s33@~{������RB�
=�s33@�{>�Q�@�  Bͨ�                                    Bxf|H�  T          @�33���@�  �����=qBх���@�{>�ff@��B�G�                                    Bxf|W�  
�          @�녿��@~�R��p���\)B�𤿇�@�z�?   @�ffB��)                                    Bxf|f&  �          @�����@|(���33��p�B�33���@��
������BÊ=                                    Bxf|t�  T          @�p��z�@}p���{����BĔ{�z�@��
���
���
B�
=                                    Bxf|�r  �          @�zἣ�
@{���(��ԣ�B������
@�z�L���#�
B��=                                    Bxf|�  �          @��=u@s33�ff���
B�{=u@�=q�\��Q�B�Q�                                    Bxf|��  T          @��ͼ#�
@s33����RB�33�#�
@����ff���B�.                                    Bxf|�d  T          @����Q�@���������B�p���Q�@�\)<��
>�z�B�=q                                    Bxf|�
  
�          @�{�L��@�(���{���RB�� �L��@�>B�\@!G�B�ff                                    Bxf|̰  "          @���@���p���{B�.��@�p�>���@���B�                                      Bxf|�V  �          @��;aG�@�{��=q��Q�B���aG�@��
>�@�ffB��q                                    Bxf|��  "          @�zᾀ  @�G�������B�(���  @��
=���?��RB���                                    Bxf|��  
Z          @��H�8Q�@r�\���z�B�z�8Q�@�녾Ǯ��(�B���                                    Bxf}H  	�          @���8Q�@n�R�G���\)B���8Q�@�
=��33��G�B��)                                    Bxf}�  T          @��R>�@y����(�����B���>�@��>aG�@@��B�k�                                    Bxf}$�  
�          @�����@c33�p���Q�B�.���@�z�(��  BĀ                                     Bxf}3:  
�          @�  �(��@`  �z���B�z�(��@�(��:�H�
=B�=q                                    Bxf}A�  �          @��
�aG�@C33�(����HB�
=�aG�@vff�������Bͣ�                                    Bxf}P�  �          @�ff��G�@8Q��7��,�
B���G�@q녿�p���(�B���                                    Bxf}_,            @����H@g
=��\��(�B�k����H@��
��G���G�B��                                    Bxf}m�  )          @�\)�u@x�ÿ�����B�Ǯ�u@�
==#�
>��B���                                    Bxf}|x  
�          @�ff��Q�@fff�����33B�Ǯ��Q�@��Ϳ���\B�ff                                    Bxf}�  "          @��=p�@a��33��RB˽q�=p�@������H��\)BȽq                                    Bxf}��  
�          @��ͿaG�@mp�������B��aG�@�G�    �#�
B̨�                                    Bxf}�j  �          @�z�G�@j�H��(���33B��G�@����\)���Bɽq                                    Bxf}�  �          @��R�=p�@j=q��Q��ۅBʸR�=p�@�(���{��p�B�33                                    Bxf}Ŷ  )          @����Q�@qG���Q���B��
��Q�@�(���\)�p��B��f                                    Bxf}�\  T          @����@p�׿�p�����B��
��@�G�=�G�?�ffB½q                                    Bxf}�  "          @�33��\)@aG���\��B����\)@�G��   �߮B��                                    Bxf}�  
Z          @��þ��@Tz��\)�ffB�Ǯ���@{��B�\�.ffB�k�                                    Bxf~ N  �          @��׿333@�G���z��}p�BǞ��333@�p�?   @ٙ�B�                                      Bxf~�  "          @�녿@  @�=q��
=�~{B�Ǯ�@  @��R?   @�B��                                    Bxf~�  
�          @�G���R@�=q��(���Q�B�8R��R@�
=>�@��HBę�                                    Bxf~,@  M          @�  �s33@w�������=qB�p��s33@�z�>#�
@��B�Ǯ                                    Bxf~:�  �          @�\)�@  @~�R��33�|��B�B��@  @��
>�@ӅBȔ{                                    Bxf~I�  
�          @�  ��@�ff����33B��)��@��\?��AzffB��f                                    Bxf~X2            @���>�@�Q콏\)�c�
B�G�>�@}p�?�=qA�\)B�Q�                                    Bxf~f�  )          @��=�\)@���>�33@�\)B�33=�\)@s�
?�(�A��B���                                    Bxf~u~  �          @�G��!G�@��R?
=q@�G�B��f�!G�@j�H@�A��B�Q�                                    Bxf~�$  "          @�  ����@�p�?\)@���B��
����@h��@�A���B�k�                                    Bxf~��  T          @r�\�.{@Z=q>�=q@�  BʸR�.{@B�\?ǮA�z�B�\)                                    Bxf~�p  
�          @x��>���@>�R@{B\)B��=>���?�ff@Z�HBw��B�33                                    Bxf~�  �          @i���aG�@_\)?@  AAp�B�.�aG�@<(�@G�B	��B��                                    Bxf~��  "          @hQ�Ǯ@Fff?�A
=qB�\)�Ǯ@*�H?�A�33B�                                    Bxf~�b  
�          @j=q���
@N{�c�
�g33B�B����
@Tz�>���@�p�B��                                    Bxf~�  �          @i���aG�@.�R@�B!�B�(��aG�?��@L��Bz�B�33                                    Bxf~�  �          @e�>.{@U�?�G�A�G�B�G�>.{@&ff@�B+��B��3                                    Bxf~�T  �          @`��>�\)@L��?�z�A�B���>�\)@�H@!G�B7p�B���                                    Bxf�  �          @X�þ�G�@2�\?�G�A�B��;�G�?�@,(�BW(�Bͮ                                    Bxf�  �          @g�����@1�?��HB�B��Ϳ���?�=q@7�BRp�B��=                                    Bxf%F  �          @e��ff@>�R?�
=A�=qB���ff@Q�@+�BA��B�u�                                    Bxf3�  
�          @l(���G�@?\)@�\B33BĸR��G�@   @@��B_(�B̽q                                    BxfB�  �          @e��W
=@=p�?�(�Bp�B�{�W
=@   @<(�B^=qB��f                                    BxfQ8  
(          @[���@0��@ ��B(�B���?�@9��Bh  B��=                                    Bxf_�  �          @Tz�=���@   @
=qB"�B�k�=���?\@<(�Bz��B�u�                                    Bxfn�  T          @c�
?�@7�@�Bp�B�u�?�?�33@<��Ba  B�z�                                    Bxf}*  "          @e�fff@(�@'�B?�HB�׿fff?�ff@P  B��)Cp�                                    Bxf��  "          @k���  ?��@@  Ba=qC
�
��  =�G�@QG�B�� C/�                                    Bxf�v  �          @n�R��R?&ff@4z�BJQ�C#����R��ff@7
=BN(�C?Q�                                    Bxf�  �          @n{�7
=?
=q@��B��C)Q��7
=��{@33B��C:�q                                    Bxf��  
�          @qG��3�
?:�H@��BG�C%Q��3�
�L��@   B&p�C8#�                                    Bxf�h  �          @o\)�<��?J=q@��B�C%��<�ͽu@�B��C5�                                    Bxf�  
�          @qG��Q�?}p�?ǮAîC#(��Q�>�33?�A�\C-�                                    Bxf�  �          @s33�Vff?��?��\A}p�C�R�Vff?Q�?�A�
=C&0�                                    Bxf�Z  T          @p���N�R?��H?�
=A���Cp��N�R?��?��
A㙚C)xR                                    Bxf�   "          @vff�P��?���?�33A���C���P��?W
=?�A���C%�{                                    Bxf��  "          @mp��I��?�=q?���A�ffC!H�I��?:�H?�\A�  C&�                                    Bxf�L  �          @k��Tz�?O\)?��A���C&G��Tz�>�\)?�G�A���C/0�                                    Bxf�,�  "          @l���S�
?:�H?���A�p�C'���S�
>8Q�?ǮA��HC0��                                    Bxf�;�  �          @j=q�P��>�{?�p�A£�C.)�P�׾k�?�  Ař�C7�3                                    Bxf�J>  "          @`���8Q���?�33A�=qC?��8Q�xQ�?Y��Az�\CF��                                    Bxf�X�  
Z          @g
=��Ϳ�G�����33CW��Ϳz�H�ff�'{CI                                    Bxf�g�  �          @i���[��s33�(���(��CCff�[��#�
�u�v{C>�
                                    Bxf�v0  �          @k��U���\�&ff�%CH�H�U�s33�������HCC�                                    Bxf���  Z          @g��N�R��
=��=q��\)CH)�N�R�8Q쿷
=��G�C@z�                                    Bxf��|  �          @e�U��\)�0���0��CF���U�L�Ϳ�ff��p�CA�                                    Bxf��"  
�          @b�\�U�Q녿:�H�>�RCA���U�   �xQ���(�C<�=                                    Bxf���  	�          @Z=q�Q녿z�5�?\)C>
�Q녾�z�^�R�l(�C9
                                    Bxf��n  
�          @n�R�h�ÿ�Ϳ   ����C<�
�h�þ��ÿ+��%�C9@                                     Bxf��  	�          @_\)�^�R��\)������C8���^�R�W
=�u�z=qC7��                                    Bxf�ܺ  T          @^{�^{<��
�L�Ϳ^�RC3��^{=#�
�#�
�333C3c�                                    Bxf��`  
�          @h���h��=u���ffC3��h��=��ͽ��Ϳ�33C2n                                    Bxf��  
Z          @j=q�i��>L�;W
=�Mp�C0��i��>������C0                                      Bxf��  	�          @i���e�?����z���G�C*}q�e�?(�ýu�h��C)��                                    Bxf�R  �          @l(��g
=?8Q�L�ͿY��C(�R�g
=?0��>L��@J=qC)#�                                    Bxf�%�  �          @g
=�c33?�>���@�G�C+�{�c33>Ǯ>��@�ffC-��                                    Bxf�4�  
�          @l(��k�>W
=>#�
@#�
C0�\�k�>�>k�@aG�C1��                                    Bxf�CD  
�          @mp��l�;L�ͼ��
���
C7��l�;8Q콸Q쿮{C6�R                                    Bxf�Q�  
�          @mp��k�>.{��z����C1G��k�>�=q�W
=�Tz�C/�)                                    Bxf�`�  �          @hQ��c�
<#�
�0���/
=C3�c�
>�  �#�
�"ffC/��                                    Bxf�o6  T          @k��dz�>8Q�(���HC1��dz�>\��\� ��C-�)                                    Bxf�}�  "          @j=q�Z=q>\)��ff��\)C1�f�Z=q?
=����  C*B�                                    Bxf���  �          @e��Mp�<#�
��  ��{C3�H�Mp�?���33���\C*��                                    Bxf��(  "          @e��C33��Q��=q��C5���C33?�Ϳ�  ��\)C)��                                    Bxf���  "          @k��&ff�u����,z�C9=q�&ff?�R�Q��&�\C&��                                    Bxf��t  
�          @n�R��0���?\)�V33CFJ=�>\�B�\�\{C)��                                    Bxf��  T          @i����G���  �H���p=qCU���G�>\)�R�\�{C.��                                    Bxf���  "          @aG��z�0���.{�M{CFJ=�z�>�z��2�\�T�C+��                                    Bxf��f  "          @^�R�p��G��{�9�CGp��p�=�G��%��EC1�                                    Bxf��  "          @l(��33��G��7��M�
CN��33=#�
�B�\�_�HC2�3                                    Bxf��  
�          @mp��%�fff�(��&�CG��%�#�
�&ff�4=qC45�                                    Bxf�X  
�          @u��%��#�
�-p��6p�CAٚ�%�>����1G��;(�C,Ǯ                                    Bxf��  Z          @qG��)���h��� ���'ffCFٚ�)��    �*�H�4�C3�3                                    Bxf�-�  "          @s�
�J=q���׿�G���p�CG��J=q�����\�G�C<aH                                    Bxf�<J  "          @w
=�5�����z���CI�q�5��\)�$z��'��C9�H                                    Bxf�J�  
�          @z=q�J�H��\)�ff�ffC9��J�H>�G��z��\)C,
=                                    Bxf�Y�  
Z          @z=q�U�>�
=��(����C,�H�U�?�����H��z�C"n                                    Bxf�h<  "          @x���P  ?
=�G���p�C)���P  ?�p��ٙ��иRC5�                                    Bxf�v�  �          @x���J�H>�(����	(�C,Y��J�H?��׿�z���RC Q�                                    Bxf���  "          @{��5�L���,(��-C5��5?Y���#33�"�C#T{                                    Bxf��.  �          @����*=q��ff�>{�?  C=��*=q?��<���=\)C'��                                    Bxf���  
�          @~�R�/\)>��6ff�8p�C1J=�/\)?����(���'C�
                                    Bxf��z  �          @����.{>B�\�;��<
=C/���.{?�z��,(��)\)C�                                    Bxf��   �          @�=q�:=q>�33�0  �,C-��:=q?�  ��R�(�C                                    Bxf���  
Z          @�=q�E�>�\)�(���!��C.���E�?������p�C��                                    Bxf��l  T          @����@  ?(��'
=�!�
C(� �@  ?�Q������
CY�                                    Bxf��  �          @�33�N{>����   �\)C,�H�N{?����{���Cz�                                    Bxf���  �          @�=q�L��=�� ����C1��L��?p�����  C#��                                    Bxf�	^  T          @����QG����Q����C6@ �QG�?(����\�	�RC(�{                                    Bxf�  "          @�  ��
>�33�<(��M�
C+h���
?���*=q�4ffC�                                    Bxf�&�  �          @�G���\)?=p��l(��C���\)?�z��O\)�X33B��                                    Bxf�5P  �          @�Q�ٙ�?=p��a��yz�C}q�ٙ�?����Fff�K�HCz�                                    Bxf�C�  "          @\)��>u�W��j�C-J=��?����G
=�P
=C�f                                    Bxf�R�  T          @����zᾅ��Z�H�j33C;G��z�?\(��Tz��_��Cn                                    Bxf�aB  �          @����=q����I���O  CB��=q>�ff�J�H�QG�C)Q�                                    Bxf�o�  �          @~{�
=�@  �P���]�HCG�
�
=>�{�U��d��C*��                                    Bxf�~�  
�          @����fff?
=q�uL�Cٚ�fff?�G��]p��q
=B�Q�                                    Bxf��4  
�          @��Ϳ   ?�ff�o\)\BݸR�   @(��G��M�B�                                    Bxf���  T          @�(��5?����u��B잸�5@\)�L���L(�B�                                      Bxf���  
�          @���L��?�G��j�H��B�녿L��@Q��Dz��J��B�ff                                    Bxf��&  T          @�{���H?���p  �~�C
���H@��L(��F��B�G�                                    Bxf���  
�          @��;��R?�(��j�H8RBȅ���R@333�;��8�B���                                    Bxf��r  
(          @�ff�u?�=q�vffp�B�ff�u@.{�H���Dp�B��=                                    Bxf��  �          @��<�?����y��aHB�<�@0���L(��Dz�B��3                                    Bxf��  
�          @��ý�G�@\��� ����B�����G�@�=q�����{B��                                    Bxf�d  "          @�ff�#�
@XQ���R�
=B�W
�#�
@\)���
��(�B�G�                                    Bxf�
  T          @�{����@H���1G��%�B��=����@vff�����Q�B��H                                    Bxf��  
�          @��?z�?�G��o\)8RB��q?z�@Q��I���Pz�B���                                    Bxf�.V  
�          @�p�?Q�?���N�R�j�\B��q?Q�@.{� ���%��B�aH                                    Bxf�<�  
�          @�33=u@:=q�.�R�,�B���=u@g���
=��z�B��                                    Bxf�K�  �          @��\��z�@AG��&ff�"ffB����z�@l(��\��
=B�#�                                    Bxf�ZH  �          @����   @\�Ϳ�\)��B�aH�   @w��333�"{B¨�                                    Bxf�h�  T          @��
�^�R@^{��(���z�B�{�^�R@z=q�G��1p�B�                                    Bxf�w�  T          @���?0��@
=�Q��TffB�33?0��@QG��Q���B���                                    Bxf��:  T          @�33>#�
@"�\�J=q�L�\B��>#�
@X���p��33B��3                                    Bxf���  �          @�z�?J=q@z��Y���c��B�G�?J=q@AG��%���B��q                                    Bxf���  T          @�
=?B�\@��\���\33B��?B�\@O\)�$z���B��                                    Bxf��,  
Z          @��?޸R?�(��\(��Z��B2�?޸R@,(��/\)�"=qBd�H                                    Bxf���  "          @��?��H@	���C33�9Q�B>��?��H@?\)�\)� 33Bc                                      Bxf��x  T          @�
=?�\)@0  �3�
�)�RB~p�?�\)@^�R��=q��(�B�.                                    Bxf��  
�          @�ff?��H@��?\)�7��BW{?��H@H���Q���ffBu�                                    Bxf���  �          @�=q?�=q@�\�Tz��I�
B@ff?�=q@=p��"�\�\)Bi
=                                    Bxf��j  "          @�G�@�?����Z=q�S�Aޣ�@�@(��8Q��)ffB/G�                                    Bxf�
  �          @���@$z�=��X���Sff@*�H@$z�?�33�L(��B�A��R                                    Bxf��  "          @�=q@#33��(��\(��TG�C�+�@#33?!G��Z=q�Q�\A_33                                    Bxf�'\  T          @�  ?h��?fff�a�B3=q?h��?�
=�E�]=qB��=                                    Bxf�6  �          @��Ϳ}p�@#33�\(��N{B�W
�}p�@^{� ����B�                                    Bxf�D�  �          @�
=�  @7
=��ff��=qC 5��  @L(��(��=qB�\)                                    Bxf�SN  �          @�=q�33@G�@
=qB��C\)�33?\@/\)B3p�C�)                                    Bxf�a�  "          @����xQ�+�@tz�B���CV���xQ���@\(�Bmz�Cq��                                    Bxf�p�  
�          @��ÿ��H>.{@n�RB��)C-E���H�s33@g
=B�Q�CU�                                    Bxf�@  T          @������?��H@e�Bt�C8R����=�Q�@q�B��C0�
                                    Bxf���  �          @�{�ff@	��@:=qB0G�C\)�ff?�
=@Z�HB[=qC��                                    Bxf���  �          @}p�����?��@EBM�HC�f����?c�
@`��B{Q�C��                                    Bxf��2  �          @�z����@
�H@H��BFQ�B��
����?���@h��Bx
=C&f                                    Bxf���  "          @�p���{?}p�@hQ�Bz{C���{�\)@p��B�z�C8��                                    Bxf��~  �          @��׿�=q>�=q@��B�� C+����=q�u@�Q�Bz�RCO��                                    Bxf��$  "          @�G��
=q?z�@|(�Bqp�C$���
=q�(�@{�Bq  CC��                                    Bxf���  �          @����?:�H@��Bx=qC {�녿   @�33B|�CA                                    Bxf��p  T          @�(��33?Y��@�G�Bt��C���33�\@��B|z�C>p�                                    Bxf�  T          @�p��G�?�ff@uBa�RC���G�>�Q�@�(�B~33C*                                    Bxf��  "          @�p����@��@E�B&�\C����?�Q�@j=qBP�
Cp�                                    Bxf� b  
�          @�(��*=q@:=q@��A��Cff�*=q@@HQ�B+ffC޸                                    Bxf�/  �          @�=q�-p�@&ff@"�\B�C)�-p�?�  @K�B2��C
                                    Bxf�=�  �          @����8Q�@��@�A�(�C+��8Q�?��@8Q�B#�Ch�                                    Bxf�LT  �          @��\�.{@�@z�B��C
(��.{?�33@:�HB*ffC�f                                    Bxf�Z�  �          @���3�
@/\)?���A��C��3�
@ff@"�\B��C=q                                    Bxf�i�  
�          @�=q�HQ�@Dz�?��HAw�C�H�HQ�@%�@33A��Cz�                                    Bxf�xF  �          @�(��P  @G�?�ffAQC+��P  @,(�?�z�A�ffCn                                    Bxf���  �          @��H�\��@8Q�?.{A	�C5��\��@#�
?�p�A���Cs3                                    Bxf���  
�          @���<��@?\)?���A��C�{�<��@@&ffB
ffC��                                    Bxf��8  �          @���I��@(Q�?�z�AǮC!H�I��?�(�@%�B33C�                                    Bxf���  �          @�ff�+�@8Q�?��RA�ffC�)�+�@��@.�RB��C�H                                    Bxf���  �          @���3�
@H��?��A�=qC�)�3�
@$z�@�A���C	�\                                    Bxf��*  
�          @����ff@^{?��RA˅B�8R�ff@0��@8��B�\Ck�                                    Bxf���  "          @�z����@XQ�@{BffB��
����@"�\@U�B8z�B��f                                    Bxf��v  �          @�(���ff@a�@!�B33B�B���ff@*�H@[�B@{B��                                    Bxf��  
�          @����ٙ�@o\)@��A�=qB�녿ٙ�@9��@Z=qB5�B��)                                    Bxf�
�  �          @�(��(Q�@C�
@��A�(�C�3�(Q�@33@A�B#��C
޸                                    Bxf�h  �          @�{�.{@G
=@�RA��C.�.{@
=@@��B��C{                                    Bxf�(  
�          @�G��   @U�@A�
=B��   @"�\@K�B'(�C�=                                    Bxf�6�  �          @������@���?�p�A��
B�Q����@U@A�BB�W
                                    Bxf�EZ  T          @����xQ�@��?�ffA|��B�aH�xQ�@|��@   A���Bϙ�                                    Bxf�T   �          @�
=��  @�  ?uA;�
B�׿�  @tz�@ffA���B�#�                                    Bxf�b�  
�          @��R��Q�@�33?�=qAW33BҔ{��Q�@xQ�@\)A�B�{                                    Bxf�qL  "          @��׿���@���?ٙ�A�  Bѣ׿���@hQ�@3�
B��B֏\                                    Bxf��  �          @�z�B�\@��R?W
=A(��B�=q�B�\@���@33A�
=B�#�                                    Bxf���  �          @��\��@�G�>��?�z�B�  ��@�=q?�33A���B���                                    Bxf��>  �          @�����R@���   ��
=B����R@�(�?:�HA�B�33                                    Bxf���  "          @��H=�Q�@�\)>�Q�@��B��{=�Q�@�ff?�=qA��HB�k�                                    Bxf���  "          @��
>\)@�33>�Q�@�Q�B��=>\)@�=q?�{A�\)B�Q�                                    Bxf��0  T          @��\<#�
@���>�@��B��)<#�
@�\)?�
=A���B��)                                    Bxf���  "          @�=q>k�@�Q�?z�@��HB�B�>k�@�?��A�G�B���                                    Bxf��|  �          @��\��33@��׾��R�}p�B�Ǯ��33@�?n{A<��B���                                    Bxf��"  
�          @��=L��@�Q쾽p���
=B��3=L��@�{?^�RA0��B��                                    Bxf��  T          @��þW
=@���<#�
>#�
B�ff�W
=@�\)?���A}�B���                                    Bxf�n  �          @�  �W
=@�Q쿮{����B����W
=@�
=�����Q�B�W
                                    Bxf�!  �          @��\���@�  ��\)�eB����@��?p��A?�
B�33                                    Bxf�/�  T          @�=q�
=q@�{?n{A<  B���
=q@���@A�  B�ff                                    Bxf�>`  T          @��>��R@�z�k��A�B�>��R@�G�?uAHQ�B��
                                    Bxf�M  "          @�33��\)@�
=?!G�@�
=B��)��\)@�z�?�ffA�=qB�                                      Bxf�[�  
�          @��?=p�@�녾8Q���RB��\?=p�@�{?�ffAQp�B�{                                    Bxf�jR  �          @�ff?�33@�{������B�{?�33@�p���\)�^�RB��                                    Bxf�x�  
�          @���?��H@����G��K�B��?��H@�33>��?�z�B��H                                    Bxf���  
�          @�ff@G�@���fff�2{Bp�@G�@��R>aG�@0  B��3                                    Bxf��D  �          @�{@G�@�녿z�H�C33B~{@G�@�>�?�{B�L�                                    Bxf���  �          @�\)?��@��Ϳ333�
�RB�
=?��@�>��H@���B�=q                                    Bxf���  T          @�
=?&ff@���>���@�(�B�33?&ff@�(�?�{A�Q�B�8R                                    Bxf��6  �          @��?��@�p�>k�@0��B�z�?��@�ff?���A�G�B�                                    Bxf���  �          @���?�R@�\)��\)�Q�B���?�R@��\?�
=Adz�B��                                     Bxf�߂  T          @�{>�{@���
=q����B���>�{@��H?0��A(�B��{                                    Bxf��(  �          @�ff?E�@�Q�Ǯ��33B��{?E�@�ff?L��A!��B�W
                                    Bxf���  �          @��@��@��<��
>L��Bs@��@z�H?�=qAV=qBp33                                    Bxf�t  "          @�z�@	��@�녿
=q��
=BxG�@	��@��?�@�z�BxQ�                                    Bxf�  �          @�?��@���+��
=B�  ?��@�(�>��H@���B�33                                    Bxf�(�  T          @�?�(�@�(��:�H��HB��)?�(�@�p�>�(�@��\B�.                                    Bxf�7f  �          @���?#�
@��\��p����
B�(�?#�
@�녾��R�xQ�B�                                    Bxf�F  �          @�z�?�G�@�33�fff�333B�?�G�@�{>��@L(�B�W
                                    Bxf�T�  "          @��\?xQ�@��R�L���$z�B��=?xQ�@��?s33A@(�B�                                    Bxf�cX  T          @��?L��@��׾W
=�(Q�B���?L��@�p�?uA@��B��=                                    Bxf�q�  T          @�\)?���@��Ϳ����]B��{?���@�������Q�B�#�                                    Bxf���  �          @�\)@G�@�Q쿆ff�N{Bq��@G�@�z�<#�
=���Bu33                                    Bxf��J  
(          @��R@@x�ÿ�z��f{Bk��@@�녾���\)Bp33                                    Bxf���  
Z          @�ff@333@^{���R���BLG�@333@o\)�\)��G�BTp�                                    Bxf���  
�          @�z�@1�@P�׿޸R��ffBFz�@1�@fff�Y���)G�BQff                                    Bxf��<  
(          @�{�#�
@���?�
=Ap  B���#�
@}p�@��A�B��\                                    Bxf���  �          @��>k�@�=q?�p�A��HB�.>k�@���@%B  B�p�                                    Bxf�؈  �          @��?+�@���?���A��B�=q?+�@���@�RA��B�=q                                    Bxf��.  T          @�=q>�
=@�p�?��AP��B��>�
=@�  @�RAܣ�B���                                    Bxf���  
Z          @�=q<#�
@���?�
=Ac\)B��q<#�
@��R@�
A�=qB��3                                    Bxf�z  �          @�=q=�Q�@���?�(�Ak�B�=�Q�@�{@ffA�{B��                                    Bxf�   
�          @��H��  @��?^�RA&=qB����  @��
@�A���B���                                    Bxf�!�  �          @��H��p�@�  ?0��A�B�녾�p�@�?���A�33B��{                                    Bxf�0l  T          @��\���R@���>�Q�@��B�� ���R@�G�?��A�B��f                                    Bxf�?  "          @��
����@��?\)@��
B�B�����@���?޸RA�  B��R                                    Bxf�M�  "          @�z�B�\@�=q�#�
��\)B��f�B�\@��R?��AE�B�Q�                                    Bxf�\^  
�          @����p�@�33�Y���   B׏\��p�@�p�>��R@j�HB�{                                    Bxf�k  T          @�(���{@��
�5�B��)��{@��>�@��
BԨ�                                    Bxf�y�  	�          @�p����R@�{�=p��
=BѸR���R@�
=>�G�@��
Bр                                     Bxf��P  
Z          @�����@��R�:�H�	�B�B����@�  >�ff@���B�\                                    Bxf���  "          @��R��{@�(�����s\)B�  ��{@�������\)B�
=                                    Bxf���  
�          @���Q�@�녿�G��;
=B�=q�Q�@��>L��@33B��)                                    Bxf��B  T          @�p��\)@��ÿaG��%�B�\)�\)@�33>��R@l(�B�(�                                    Bxf���            @��H��z�@�33��=q����B�G���z�@��þ.{� ��B�                                      Bxf�ю  '          @�(����@��׿0���ffB�#׿��@�G�>��H@�33B�\                                    Bxf��4  �          @�33��{@�녾�33��(�B��ᾮ{@�  ?Tz�AffB�{                                    Bxf���  �          @�33��\@��������aG�B�33��\@�
=?aG�A&�HB�aH                                    Bxf���  '          @��
��=q@������Ϳ�(�B�p���=q@�?��AL(�B���                                    Bxf�&  �          @��\>��H@��R�fff�+�B�33>��H@�G�>�=q@K�B�aH                                    Bxf��  �          @��\?�@�����\)�U��B�B�?�@�{<#�
>.{B�\                                    Bxf�)r  �          @�Q�=�\)@�z�J=q�33B�8R=�\)@�ff>�33@�
=B�=q                                    Bxf�8  T          @�zῡG�@��ÿ�G��o\)B�.��G�@�ff���ǮB��                                    Bxf�F�  �          @��Ϳ�G�@�(�����X��B߸R��G�@��ýL�Ϳ#�
B�u�                                    Bxf�Ud  �          @�(����@��H����Q�B����@�\)���
����Bߊ=                                    Bxf�d
  T          @�{��(�@���  ���\B�Ǯ��(�@��������{B�{                                    Bxf�r�  �          @��
<#�
@�ff����z�B��<#�
@��ÿL�����B��3                                    Bxf��V  �          @�(�>L��@�Q�@  ��
B�33>L��@���>���@�\)B�=q                                    Bxf���  T          @�(��B�\@�z�h���/
=Bƙ��B�\@�
=>aG�@&ffB�G�                                    Bxf���  �          @�z�<�@�녿=p���B�aH<�@��H>�
=@�ffB�aH                                    Bxf��H  �          @�>\)@����z�H�9�B��q>\)@���>8Q�@�B���                                    Bxf���  �          @��;��H@�p�������(�B��f���H@�33�W
=�{B�p�                                    Bxf�ʔ  �          @���@�(���Q���ffB�#׾�@��Ϳ����ffB���                                    Bxf��:  �          @�p�>��@����  ���HB�p�>��@�z᾽p�����B���                                    Bxf���  �          @�  >�\)@�33��z����
B��>�\)@�p��G��\)B�aH                                    Bxf���  �          @�ff�k�@�������홚B�#׾k�@�\)��������B��\                                    Bxf�,  �          @�ff�L��@�p��(����B�B��L��@�녿����N�\B���                                    Bxf��  T          @��\?W
=@�z������B�33?W
=@��8Q��
ffB��\                                    Bxf�"x  �          @�=q>�=q@��
�����G�B��>�=q@��\����  B�B�                                    Bxf�1  �          @��H�L��@����{��p�B�G��L��@����`z�B�(�                                    Bxf�?�  �          @��\��33@��\��ͮB�L;�33@�ff���
�F{B��{                                    Bxf�Nj  T          @����.{@�G����R����B�Q�.{@��׾�������B��                                    Bxf�]  �          @�녾Ǯ@�ff�޸R����B��Ǯ@�\)�+�� ��B�W
                                    Bxf�k�  T          @��Ϳ}p�@�(�����p�B�녿}p�@�=q��{���B�p�                                    Bxf�z\  �          @��H��=q@�G���
=���HB�G���=q@��
�fff�+\)B�Q�                                    Bxf��  T          @��\�:�H@���������B�LͿ:�H@�  ��33��Q�B�\)                                    Bxf���  �          @���p��@���(���
=B�8R�p��@�Q쿹����B˨�                                    Bxf��N  �          @��z�H@����#33��33B�W
�z�H@�  ������=qB̊=                                    Bxf���  �          @��ÿ��@����'����B����@�G��У���33B��                                    Bxf�Ú  �          @��
��=q@�녿�  �o�B����=q@�\)�u�5B�aH                                    Bxf��@  T          @�33�@  @�녿���|��B�Ǯ�@  @���u�5B�{                                    Bxf���  �          @�33>�@�
=��{��33B��3>�@�
=�����(�B�W
                                    Bxf��  �          @��
���@�p���p����\B�(����@�  �n{�1�B�L�                                    Bxf��2  �          @���\@��R��  ���B۽q�\@���B�\��B�aH                                    Bxf��  T          @����33@��׿�\)����B�G���33@��׿(���Q�B��                                    Bxf�~  �          @�(����@�\)������=qB�����@�G��Y��� Q�B�u�                                    Bxf�*$  �          @�z�^�R@�  �\)��  B�33�^�R@��Ϳ�p��hQ�B�G�                                    Bxf�8�  T          @��Ϳ�@��\�(�����B��
��@�=q��z���z�B�B�                                    Bxf�Gp  T          @��
�\)@u��=p����BĨ��\)@�z�������B=                                    Bxf�V  �          @�z�\)@���(Q���HB��
�\)@�G���z���p�B�(�                                    Bxf�d�  �          @��׿���@`  �-p��p�B�  ����@��׿�\)����B۳3                                    Bxf�sb  �          @���5@`  �z��مC  �5@{���p����B���                                    Bxf��  T          @���@E��P  �   B�aH�@n�R��R��p�B�8R                                    Bxf���  �          @�33��  @h���L���33B��Ϳ�  @�Q���
�أ�B���                                    Bxf��T  �          @�33�=p�@L���y���GQ�B�#׿=p�@\)�E�=qB�#�                                    Bxf���  �          @��H��p�@R�\�k��9ffB�녿�p�@����7
=�	
=B��                                    Bxf���  �          @����G�@e��Z=q�'�HBڮ��G�@���"�\��RB���                                    Bxf��F  �          @���˅@O\)�l(��6z�B�p��˅@~{�8Q����B�                                    Bxf���  �          @�33����@H���k��8�B��ÿ���@w��9���
�RB��H                                    Bxf��  �          @�ff�޸R@Y���G
=�z�B�8R�޸R@�  ��\��33B�\                                    Bxf��8  �          @��׿��@�G��333���B�#׿��@��\>��R@p  B��f                                    Bxf��  �          @�Q�G�@���Tz��!��B�Q�G�@�>B�\@ffB�
=                                    Bxf��  �          @�G��\(�@�\)��z���Bɽq�\(�@��������HB���                                    Bxf�#*  �          @�(��B�\@��Ϳ������HBǣ׿B�\@�
=�s33�333B�Q�                                    Bxf�1�  �          @���333@���\)��ffB���333@�����
��33B�L�                                    Bxf�@v  �          @���Q�@qG��B�\�  B̨��Q�@��H�	���ϮBɀ                                     Bxf�O  �          @��Ϳ=p�@j�H�I��� Q�B��)�=p�@�Q�����=qBǸR                                    Bxf�]�  �          @����\@mp�����
=B�\��\@�녿�
=�h��B�G�                                    Bxf�lh  �          @���I��@Mp���ff����C}q�I��@aG������K\)C�{                                    Bxf�{  �          @�z��-p�@U��33��p�C!H�-p�@o\)���
��=qB��R                                    Bxf���  T          @�p���@S�
�@  ���B����@xQ��p��֣�B�(�                                    Bxf��Z  �          @�p��G�@X���;��p�B��G�@|(�������B�L�                                    Bxf��   �          @�(��ff@X���4z��Q�B��ff@z=q���ģ�B�\)                                    Bxf���  �          @��޸R@c33�:�H���B�LͿ޸R@��H���33B��                                    Bxf��L  �          @�(����@Vff�C�
��B�\)���@z�H�G���ffB��                                    Bxf���  �          @��
��@QG��G
=��B��
��@w
=���=qB�.                                    Bxf��  �          @�33���H@J�H�E��33B�W
���H@p�������B�
=                                    Bxf��>  �          @�(���p�@G
=�Tz��,
=B�=q��p�@p  �%�� �B��                                    Bxf���  �          @�z���\@333�Vff�0�B�=q��\@\���+��
=B�L�                                    Bxf��  T          @���J=q@�
�9���{C� �J=q@8Q�����
C	�f                                    Bxf�0  T          @��H��{@0���hQ��B��B��\��{@]p��=p��33B��)                                    Bxf�*�  �          @�G���\)@)���s�
�S��B���\)@Y���J=q�%��B�aH                                    Bxf�9|  �          @�Q쿣�
@!G��q��T�HB��Ϳ��
@P���J=q�(G�B���                                    Bxf�H"  
�          @��R��G�@#33�x���[p�B�G���G�@S�
�QG��-ffB��                                    Bxf�V�  �          @�(���G�@|(������33B�\��G�@��.{��Bី                                    Bxf�en  T          @��\��@fff��\)��G�B����@z=q��=q�[�B��                                    Bxf�t  T          @�녿��H@~�R�������B�\)���H@�p�����
=Bڞ�                                    Bxf���  �          @��Q�@>�R��\����C ���Q�@U��\)��\)B��                                    Bxf��`  �          @�p��*�H@Q��K��+�CQ��*�H@0  �*=q�  C
                                    Bxf��  T          @�{�1�@&ff�.�R��\C���1�@G
=�Q����C��                                    Bxf���  �          @��R�5@:�H�ff��C+��5@Vff��Q���{CJ=                                    Bxf��R  
�          @�\)�<��@Dz��G�����C�{�<��@Z�H��=q��p�C��                                    Bxf���  �          @�
=�Q�@C�
��Q����C	  �Q�@R�\�E���C��                                    Bxf�ڞ  
�          @��R��\@Dz��C�
�#Q�B���\@h���
=��B��H                                    Bxf��D  �          @�p����H@Dz��7��  B�8R���H@e�
�H��\)B�=q                                    Bxf���  �          @��R�@HQ��%��B��=�@fff�������B�
=                                    Bxf��  �          @���-p�@O\)�����G�C���-p�@g
=����(�B��3                                    Bxf�6  T          @���C�
@>�R�33�̏\C�q�C�
@U�������
C}q                                    Bxf�#�  "          @�\)�A�@B�\��(���33C�f�A�@XQ쿦ff�\)Cٚ                                    Bxf�2�  �          @�ff�;�@AG������HC5��;�@XQ쿵���\C�3                                    Bxf�A(  T          @���L(�@AG���Q����
C�\�L(�@S33���
�I��C                                      Bxf�O�  �          @����Q�@1G��33��z�C��Q�@HQ쿸Q�����CE                                    Bxf�^t  �          @��H�A�@Q��G�� �C��A�@.{�'
=��
C
�                                    Bxf�m  �          @�Q��:=q@0���!G�� �C� �:=q@N{��33��
=C)                                    Bxf�{�  �          @�G��$z�@@  �+���RC���$z�@_\)� ����33B�                                    Bxf��f  �          @�Q���@>�R�@  �G�B�Q���@a��z���  B�W
                                    Bxf��  �          @�G��   @1G��>{�(�C�q�   @Tz����33B��
                                    Bxf���  T          @����'�@\)�Fff�"�C}q�'�@Dz��!�� 33C}q                                    Bxf��X  �          @�G��:�H@#33�0���  C
���:�H@C�
�(��ڣ�C��                                    Bxf���  �          @��H�C33@���8Q��p�C�)�C33@<(������C!H                                    Bxf�Ӥ  T          @��\�W�?����;���RC�
�W�@Q��   ���C�                                     Bxf��J  �          @����p  ?�\)�4z����C�3�p  ?�z���R���C��                                    Bxf���  �          @�(��z�H?�{�,���  C$:��z�H?У���H��
=Cff                                    Bxf���  T          @�{��  ?c�
�0  �z�C'n��  ?�
=� ������C L�                                    Bxf�<  T          @���:=q?���6ff��Cc��:=q@���H�{CO\                                    Bxf��  �          @�(����@<���Mp��/��B�  ���@b�\�#33��RB��f                                    Bxf�+�  �          @�zΌ��@/\)�]p��@p�B�3����@XQ��5��G�B�W
                                    Bxf�:.  �          @��Ϳ�ff@   �^�R�AB��\��ff@I���9���z�B�=                                    Bxf�H�  
�          @�ff��
?��R�����lC����
@��l���O��Cu�                                    Bxf�Wz  �          @��
�˅@ ���u��`�HC h��˅@/\)�Vff�:Q�B�Q�                                    Bxf�f   �          @�녿���@{�aG��K(�B��Ϳ���@HQ��<(��!�HB�q                                    Bxf�t�  �          @�����Q�@?\)�Mp��3��B߀ ��Q�@dz��"�\�  B��f                                    Bxf��l  T          @��׿��@W��3�
�\)B�k����@w
=���ۙ�B�33                                    Bxf��  �          @�G���33@&ff�`���L{B㞸��33@P  �:�H� ��B��)                                    Bxf���  �          @�G���ff@��ÿ�G���(�B�{��ff@�{��p���Q�B���                                    Bxf��^  �          @�33��
=@��H��Q���Q�B�  ��
=@�G��
=q����B��                                    Bxf��  T          @����p�@�
=��(�����B�
=��p�@�\)�W
=�)p�B�u�                                    Bxf�̪  �          @���=�@�z�?ǮA��RB��=�@�\)@p�A�
=B���                                    Bxf��P  �          @��>L��@�G�?8Q�A��B�L�>L��@���?�z�A��\B�\                                    Bxf���  �          @�z�\)@���?uA6{B�Q�\)@��?�33A��B��=                                    Bxf���  T          @�G�>�  @�(�?�=qAQ�B���>�  @��\?�p�A��HB�\)                                    Bxf�B  �          @�>��@�z�>Ǯ@���B��H>��@�
=?��A�{B��                                     Bxf��  �          @�{=u@�=�G�?�z�B�k�=u@�=q?�ffAN=qB�aH                                    Bxf�$�  �          @�
=�8Q�@�p�?   @�=qB�\)�8Q�@�\)?�A�Q�B��=                                    Bxf�34  T          @�=q>u@�G�>\@���B�33>u@��
?���A�B���                                    Bxf�A�  �          @�(��L��@��H>�Q�@��B�(��L��@�p�?��A{33B�33                                    Bxf�P�  �          @��H����@��Ϳ�����\B�ff����@��H���H��B�Q�                                    Bxf�_&  �          @�=q����@����ff��\)B�
=����@�{��  ���HB�u�                                    Bxf�m�  �          @���\)@c33�:=q��
B�  �\)@�������33B�Ǯ                                    Bxf�|r  T          @���@�
=������33B���@��׿�=q�T��B�                                      Bxf��  �          @��Ϳ
=@����p����B�G��
=@��R��\)�^�HB�(�                                    Bxf���  �          @�ff����@|���!���\B������@��
�ٙ���(�B��
                                    Bxf��d  �          @���=p�@�Q쿞�R��Bǽq�=p�@�p���p����B�
=                                    Bxf��
  �          @��ÿ��R@��?
=q@�=qB�B����R@���?�\)A��RB�Q�                                    Bxf�Ű  T          @�녿��
@���?���AZ=qB�uÿ��
@��H?�p�A���B�#�                                    Bxf��V  �          @�G��У�@��
?��
AF�\B���У�@��\?�\)A�=qB�p�                                    Bxf���  "          @�p�����@�33>���@��B������@�{?�  A~{B�k�                                    Bxf��  T          @��
��33@�=q?�{A�=qB�.��33@g�@(Q�B�
B�                                     Bxf� H  �          @�=q�  @xQ�?�  A���B�.�  @]p�@�RA�(�B�
=                                    Bxf��  
(          @��H���@z=q?��
A�{B��f���@a�@G�A�z�B�8R                                    Bxf��  
�          @��
�(�@|(�?�G�A�=qB�{�(�@dz�@��A�Q�B���                                    Bxf�,:  "          @��H�(�@�z�?��AK\)B��
�(�@vff?���A�33B�Q�                                    Bxf�:�  �          @��H�޸R@��>�ff@��B�Q�޸R@�=q?�=qA~ffB��H                                    Bxf�I�  �          @�\)��@��?p��A7�B����@y��?޸RA��RB���                                    Bxf�X,  	�          @�{���@n{?��A��RB����@X��@ ��A�G�B��R                                    Bxf�f�  
�          @����%@x��?W
=A#\)B�=q�%@i��?˅A�Q�B���                                    Bxf�ux  T          @�  �G�@��?(��@�{B����G�@y��?���A�Q�B�\)                                    Bxf��  T          @�G��{@�ff=#�
>�(�B����{@��?\(�A&ffB���                                    Bxf���  
�          @�=q�@�=q�����\)B���@���?�@�B��)                                    Bxf��j  �          @�=q�G�@��\����p�B�{�G�@�33>�{@�=qB��                                    Bxf��  �          @�=q��ff@��\��\)��ffB�LͿ�ff@��׾��H��z�B���                                    Bxf���  T          @��\�У�@�zῈ���Mp�B��У�@�Q�B�\�\)B۽q                                    Bxf��\  �          @��Ϳ\@�(��������B�ff�\@�33�.{� (�Bخ                                    Bxf��  �          @�G����\@���33�ظRB��H���\@�p���z����B�k�                                    Bxf��  �          @�z���@�=q�B�\�ffB��Ϳ��@��
>u@%�B�u�                                    Bxf��N  �          @�=q��33@�{����7�B�  ��33@�����\)�E�B�=q                                    Bxf��  T          @��\��Q�@�G�������BՅ��Q�@����Q��p�B��f                                    Bxf��  �          @�녿�p�@�G���{��33BиR��p�@�녿c�
��B�=q                                    Bxf�%@  �          @�G���z�@�����
=��(�BԞ���z�@�G��5��ffB�.                                    Bxf�3�  �          @��Ϳ�  @�33�����=��B�uÿ�  @�
=��Q�k�B��)                                    Bxf�B�  "          @����ff@��\�E�� ��B�\)��ff@�(�>�\)@>{B�(�                                    Bxf�Q2  
�          @�33��p�@��׿�\)�mB�p���p�@�{��p��{�BΏ\                                    Bxf�_�  
�          @����33@�Q쿵�t��B�(���33@�ff��
=��ffB��                                    Bxf�n~  
Z          @�
=����@�녿��H�x��B�����@�  ��ff��\)B��                                    Bxf�}$  �          @�Q쿱�@�33�����Bҏ\���@�녿����
B�p�                                    Bxf���  �          @��H���H@�z��G����B�.���H@�����\�R�RB�{                                    Bxf��p  T          @�33���R@����.{���HBף׿��R@��H��G���
=BԸR                                    Bxf��  �          @�p���\@�(����H�R=qB�Ǯ��\@��׾u�#�
Bڽq                                    Bxf���  "          @�
=��33@��Ϳ�R�ٙ�B����33@�>�33@vffB��                                    Bxf��b  
Z          @�(����R@�33�G��p�BԨ����R@���>k�@�B�W
                                    Bxf��  T          @�����@�{�5���B�G�����@�\)>��R@P  B�
=                                    Bxf��  
�          @�ff��(�@�{�E��=qBӳ3��(�@��>�  @*=qB�ff                                    Bxf��T  T          @�ff��G�@����z��E�B��ÿ�G�@�������ffB�.                                    Bxf� �  
�          @�{����@��Q��
ffB�Ǯ����@��>W
=@��B�u�                                    Bxf��  T          @�ff��
=@�{�\(����B��
��
=@�  >.{?��
B�u�                                    Bxf�F  
�          @��R��
=@������7�B��ῷ
=@��׽u�&ffB�\)                                    Bxf�,�  T          @�ff��=q@�녿�G��U�B֏\��=q@��R��  �%B՞�                                    Bxf�;�  �          @����Q�@�p����H��B����Q�@���8Q���\B�W
                                    Bxf�J8  T          @�ff��=q@�33��33��
=B�#׿�=q@�(��k��
=B�L�                                    Bxf�X�  
Z          @��Ǯ@��
��ff��G�B׊=�Ǯ@�(��Q��33B��)                                    Bxf�g�  T          @�Q��(�@�=q�����RB�(���(�@�(���ff�/�B�                                      Bxf�v*  �          @�G����
@�
=��
��z�B�8R���
@��\����`(�Bڊ=                                    Bxf���  "          @�
=�У�@��R��R���B���У�@������\�W�
B׳3                                    Bxf��v  "          @�(��Ǯ@�ff�   ��\)Bڳ3�Ǯ@�33�˅��=qB׳3                                    Bxf��  �          @��׿���@r�\�Q���B��f����@�33��R��33Bۏ\                                    Bxf���  
�          @�G��Ǯ@�z��:=q�33B�aH�Ǯ@�(��33���
B�W
                                    Bxf��h  �          @�Q��  @����,(����Bڳ3��  @��R��ff��z�B�Q�                                    Bxf��  �          @�G���ff@�ff�333� �B܅��ff@�p���
=��p�B�                                    Bxf�ܴ  "          @��\��Q�@�Q��C33�33B���Q�@����{���HB���                                    Bxf��Z  
�          @�����@��H�0����B�녿��@�G���{���B�
=                                    Bxf��   �          @�z�Ǯ@�����
���B�8R�Ǯ@�Q쿨���W�
B���                                    Bxf��  "          @�
=��Q�@�ff�%��\)B�{��Q�@��������B�{                                    Bxf�L  �          @�33��p�@��\�;���p�B�
=��p�@�녿�p���Q�B�G�                                    Bxf�%�  T          @�\)��ff@���Vff�G�B�#׿�ff@�{��R��Q�B�\                                    Bxf�4�  �          @�33���H@��
�1G���\)B�
=���H@��\��\)��ffB�=q                                    Bxf�C>  "          @�
=�k�@�=q�C�
�{B�ff�k�@��H�p���(�Bʙ�                                    Bxf�Q�  T          @���(��@N�R����KB�{�(��@}p��Vff��B��                                    Bxf�`�  
�          @��R�
=@I������P{B�B��
=@x���Z=q�#�B�G�                                    Bxf�o0  �          @��\��@Tz��u��D=qB�p���@�  �G
=�=qB��                                    Bxf�}�  �          @�(�=��
@!G�����o�B�=q=��
@Tz��q��B�HB�(�                                    Bxf��|  
�          @�p��O\)@�  �J�H�ffB���O\)@�G�����Q�B�G�                                    Bxf��"  	�          @�z�B�\@��\�A��z�B��ÿB�\@�33����(�BƮ                                    Bxf���  "          @��;�ff@p  �`  �+{B�Ǯ��ff@���,�����\B��H                                    Bxf��n  �          @����\)@\���n{�<ffB�W
��\)@���>{�G�B���                                    Bxf��  	�          @�33=�\)@b�\�h���7�B��=�\)@�{�8Q��
  B�
=                                    Bxf�պ  "          @�G�>\)@g
=�^{�/�B��\>\)@�
=�,���ffB�33                                    Bxf��`  �          @��H��\)@s�
�W
=�%�B��ý�\)@����#33��RB��3                                    Bxf��  
�          @��
�#�
@|���P  ��RB��#�
@�Q��=q���B��f                                    Bxf��  
�          @��\=�\)@`���h���8{B��\=�\)@���8Q��
�HB��                                    Bxf�R  �          @��>�ff@U�p���@G�B��=>�ff@����A��ffB��                                    Bxf��  �          @��?O\)@A��|(��M��B�  ?O\)@p  �QG��!��B��                                    Bxf�-�  "          @�Q�?�  @ ����ff�gG�B~33?�  @U��w��=�B���                                    Bxf�<D  �          @�{?�@!G���z��f��B�#�?�@U��s�
�<��B�G�                                    Bxf�J�  T          @��?�
=@.�R�~�R�LBip�?�
=@\���W��%  B�(�                                    Bxf�Y�  "          @�33?��\@/\)��33�VQ�B�G�?��\@`  �^�R�,\)B�
=                                    Bxf�h6  �          @�(�?L��@
=��
=�sG�B��=?L��@L(��z=q�G�RB��)                                    Bxf�v�  �          @��?s33@$z�����gQ�B�k�?s33@XQ��qG��<{B��{                                    Bxf���  �          @�{?h��@����Q��q�\B�k�?h��@O\)�|���F\)B���                                    Bxf��(  "          @��H?&ff@33���
�B���?&ff@:�H���
�XG�B�                                      Bxf���  
�          @���?
=?�(���p�u�B�.?
=@'
=����g�B��                                    Bxf��t  
�          @���>��@=q���
�r�
B��)>��@N{�s�
�E��B�\)                                    Bxf��  �          @�G�?0��@�����B�z�?0��@<(������T\)B���                                    Bxf���  �          @��?xQ�?����{(�Bv33?xQ�@+���  �`��B�B�                                    Bxf��f  �          @�=q?G�?�Q����HW
Bb�\?G�@���Q��}��B��{                                    Bxf��  �          @���?^�R?������
=B_?^�R@p���ff�wp�B�                                      Bxf���  T          @���?��H?aG���G�W
B��?��H?������\Ba�                                    Bxf�	X  �          @���?�ff>�G����Hk�A��
?�ff?�������B<                                      Bxf��  T          @�33?���B�\��p�\)C��H?��?J=q���k�A��R                                    Bxf�&�  �          @��\?�zᾳ33��33�C��?�z�?�R���\�{A�Q�                                    Bxf�5J  �          @�\)?���L����\)p�C�1�?��>#�
��G���@ᙚ                                    Bxf�C�  �          @��R?�녿&ff��{��C�xR?��>�����\)�{A@��                                    Bxf�R�  �          @�=q?�
=�(������qC��?�
=>��R��33\)ADQ�                                    Bxf�a<  �          @��?�녿}p���Q���C�/\?�논����
33C�xR                                    Bxf�o�  "          @���?c�
����#�C�^�?c�
?
=q��B�A��H                                    Bxf�~�  �          @�G�?�  �.{��(�(�C��{?�  >��R��p��A�
=                                    Bxf��.  �          @�  ?���!G����HaHC��)?��>�33���
.A��R                                    Bxf���  �          @�?����������
C��
?���?(������3A���                                    Bxf��z  �          @��@   >��H��(��iffA0  @   ?�ff�|���Z  A�{                                    Bxf��   �          @�ff?�z�>k���ffB�A  ?�z�?�z�����W
Bz�                                    Bxf���  �          @�ff>�
=>������¨ffA�>�
=?��R���(�B�\)                                    Bxf��l  �          @�
=?!G�>�����(�¢��B�\?!G�?�\)��ffu�B���                                    Bxf��  �          @��>Ǯ>���33¥�
BHG�>Ǯ?�
=�����B���                                    Bxf��  �          @��?\?0�����G�A£�?\?�����Q��{Q�B8�                                    Bxf�^  �          @�\)@3�
>��i���Q��@*=q@3�
?\(��c33�I��A�z�                                    Bxf�  T          @��@U��33�U�3z�C��@U>����U�3�\@�{                                    Bxf��  T          @���@\��?��P���+�HA��@\��?��E�� �HA�
=                                    Bxf�.P  �          @���@N�R?����P���,\)A��@N�R?����;��\)A��                                    Bxf�<�  �          @�G�?�{?�ff�q��l
=B�
?�{@�\�\(��M{B>\)                                    Bxf�K�  �          @��Ϳ(�?����� �fC&f�(�?��R��33p�B�                                    Bxf�ZB  �          @���8Q�?������)CQ�8Q�?��R��=qL�B�W
                                    Bxf�h�  �          @�(���=q?����\¥#�B��
��=q?�����
�B���                                    Bxf�w�  �          @��
�E�?O\)��\)� C�
�E�?�  ���R�RB�k�                                    Bxf��4  T          @��\�J=q?����(�8RB�=q�J=q?�p���=q�33B�ff                                    Bxf���  �          @��H�xQ�?ٙ���p�G�B�8R�xQ�@#�
�\)�]��B�Q�                                    Bxf���  �          @�=q�5?�\)��
=��B��5@   �����eG�BӅ                                    Bxf��&  �          @�녾�z�?�\)����=B��쾔z�@.�R�|(��\�B�B�                                    Bxf���  �          @��;���@���z���Bɀ ����@>{�w
=�P�B�p�                                    Bxf��r  �          @�
=��@ ����  33B�ff��@5�p  �SffB�=q                                    Bxf��  T          @���{@����  �sz�B�G���{@>{�^{�DB��)                                    Bxf��  �          @��R>#�
?���������B��H>#�
@1��r�\�V�
B���                                    Bxf��d  �          @�ff>�=q?�����=q�B�Q�>�=q@*�H�w
=�\�B��\                                    Bxf�

  T          @�ff>u?޸R���(�B�B�>u@%�z=q�aG�B�k�                                    Bxf��  T          @�
=?�{����{G�C���?�{>���{8RA�33                                    Bxf�'V  �          @�z�?��W
=��{�C�?�?5��(�B�A�G�                                    Bxf�5�  �          @�ff?��R��R����� C��f?��R>�=q�����A%�                                    Bxf�D�  �          @��H@!G�<��
���H�i�?�\@!G�?\(��\)�a\)A��                                    Bxf�SH  �          @�  @:�H���P  �-�HC�q@:�H�����c�
�C{C��3                                    Bxf�a�  T          @���@C�
�У��QG��-��C��q@C�
�h���b�\�?�C��=                                    Bxf�p�  T          @�Q�@L(���z��G
=�#�C�7
@L(��z�H�X���5��C�k�                                    Bxf�:  �          @���@Q녿˅�Dz�� �\C�3@Q녿k��U��1��C�.                                    Bxf���  T          @��@Vff����J=q�"(�C��\@Vff�Y���Z=q�2(�C���                                    Bxf���  T          @��H@O\)��ff�Vff�/\)C�3@O\)���b�\�<��C��
                                    Bxf��,  �          @���@G����R�X���5
=C�'�@G���\�dz��A��C�\)                                    Bxf���  �          @��@?\)�����Z=q�7�C��@?\)�#�
�g��G33C��)                                    Bxf��x  
�          @�{@J�H�#�
�Y���9�
C�@ @J�H=��
�]p��>�?�                                    Bxf��  �          @�@<�ͿO\)�_\)�C33C�K�@<�ͽ�\)�e��JG�C�\)                                    Bxf���  �          @�ff@@�׿0���c33�C�
C�� @@��=u�g
=�H�H?�
=                                    Bxf��j  �          @���@G
=�@  �XQ��:\)C�9�@G
=���]p��@Q�C���                                    Bxf�  �          @��@<�ͿJ=q�\(��A�
C��@<�ͽL���a��H��C��f                                    Bxf��  T          @��\@'
=���
�o\)�\{C�#�@'
=?5�j�H�VQ�Ar=q                                    Bxf� \  T          @�=q@*�H>��H�j�H�U��A&=q@*�H?�  �^�R�F�HA�ff                                    Bxf�/  �          @���@.{>L���p  �W�
@���@.{?}p��hQ��M�A���                                    Bxf�=�  �          @���@1G�>W
=�n�R�Uz�@��
@1G�?}p��fff�K33A��                                    Bxf�LN  �          @�33@5�>L���g��O��@�33@5�?u�_\)�E�
A�(�                                    Bxf�Z�  �          @�(�@1G�>W
=�l(��T{@��@1G�?}p��dz��I�
A��R                                    Bxf�i�  �          @��
@'
=>�  �s33�]��@���@'
=?�ff�j�H�R  A���                                    Bxf�x@  �          @��@#33>8Q��u��a�@�33@#33?}p��mp��V=qA�                                    Bxf���  "          @�33@   >8Q��w
=�c�H@�ff@   ?�  �n�R�X�RA�{                                    Bxf���  
�          @�=q@!G�>���s33�a33@�33@!G�?���j=q�T�HA�ff                                    Bxf��2  "          @���@%�<#�
�o\)�]ff>�\)@%�?O\)�i���U�RA��H                                    Bxf���  
�          @�(�@\)?E��{��k��A�33@\)?˅�j�H�T�B33                                    Bxf��~  
�          @��
@p�>u��G��t��@��@p�?����y���f�AҸR                                    Bxf��$  
�          @��@�
>��R����}�
A	p�@�
?�Q���  �m�A�
=                                    Bxf���            @�{?��H>B�\��  W
@��R?��H?������u�\A�G�                                    Bxf��p  �          @��?��ýL�����
�RC�4{?���?c�
����L�A�                                    Bxf��  �          @�G�?�ff>��
��p��A!G�?�ff?�G���  �z\)B�                                    Bxf�
�  �          @�=q?�Q�>�p���  {AD  ?�Q�?�=q��=q�~\)B                                    Bxf�b  �          @���?�G�?�p���{�{��Bz�?�G�@ff�u��Y��BH33                                    Bxf�(  T          @�  ?��?������u(�B
  ?��@
=q�q��S��BC�H                                    Bxf�6�  �          @�=q@�?��\��
=�t�HA�33@�?�z��y���XQ�B*(�                                    Bxf�ET  �          @�Q�@�\>�G�����  AD��@�\?�{��=q�l
=B�
                                    Bxf�S�  �          @�
=@�>u���x��@��
@�?�������j  A�
=                                    Bxf�b�  �          @���@��?���{��f��A�z�@��?��H�e�J33B&�
                                    Bxf�qF  �          @�33@��?n{�{��j�RA�=q@��?�G��hQ��PBQ�                                    Bxf��  �          @�=q?�33>����(���AA?�33?�ff�|(��n�B	ff                                    Bxf���  �          @��
?��=�G���{Q�@Z�H?��?��
��=q�xffA�                                    Bxf��8  �          @�ff@녽�Q���
=�=C��\@�?Tz���z��x=qA�{                                    Bxf���  �          @�=q@J�H@{�1G��G�B��@J�H@1G��{���B$��                                    Bxf���  �          @���@0  @5�+����B7��@0  @W
=���R��\)BJ                                    Bxf��*  �          @��@4z�@��<����\BG�@4z�@:�H�����
B8
=                                    Bxf���  �          @�=q@p����
�s�
�c��C�K�@p�?��r�\�a�\AA�                                    Bxf��v  �          @�33@333?O\)�b�\�JG�A�G�@333?����Q��6ffA��                                    Bxf��  �          @�
=@P��?���4z��33A��
@P��@p�����B��                                    Bxf��  �          @��@?\)?��N�R�-�A�p�@?\)@�1��
=Bff                                    Bxf�h  �          @�Q�@@  ?�(��N{�+�A�
=@@  @���0  ��B\)                                    Bxf�!  �          @��H@.{?�  �Z�H�B��A��
@.{?�p��C33�(��B33                                    Bxf�/�  �          @��@(�?h���n{�e��A���@(�?�(��[��K�BG�                                    Bxf�>Z  �          @�ff@ff?�z��e�WffAҏ\@ff?�
=�O\)�;(�B�R                                    Bxf�M   �          @�=q@!G�?�p��g
=�P�A��H@!G�@ ���O\)�4�
BQ�                                    Bxf�[�  �          @���@!G�?���vff�Vp�Aۙ�@!G�@���]p��9\)B!33                                    Bxf�jL  �          @��@!G�?�
=�tz��W�A�ff@!G�@   �]p��<ffB                                      Bxf�x�  �          @��@/\)?���u�Qz�A�=q@/\)?�(��_\)�8  B�                                    Bxf���  �          @��H@.�R?�Q��s�
�OA���@.�R@G��\(��5z�B�                                    Bxf��>  �          @�(�@/\)?��u�P�HA��H@/\)@   �^�R�6�HB                                    Bxf���  �          @�(�@*=q?aG���  �[�RA��@*=q?�G��l���DB��                                    Bxf���  �          @��@#33?}p���=q�`��A�p�@#33?���p  �G(�B��                                    Bxf��0  �          @�(�@,(�?Tz��\)�[(�A��
@,(�?�(��l���D�
B�                                    Bxf���  �          @��
@,(�?J=q�~�R�[ffA�\)@,(�?��l(��E�A�\)                                    Bxf��|  �          @���@333>��~�R�YA�R@333?�\)�p���Ip�A�(�                                    Bxf��"  �          @��@:�H>��y���S33A\)@:�H?���l(��C�
Aď\                                    Bxf���  �          @�z�@8��?!G��w��R��AD��@8��?�  �hQ��@z�A��                                    Bxf�n  �          @�p�@?���z�H�_=qB��@@   �\(��:33BH33                                    Bxf�  �          @��R@33?u����{�\Aȏ\@33?�Q����\�]G�B.                                      Bxf�(�  �          @��
@33?�(���
=�r{A���@33@	���u��P��B9�                                    Bxf�7`  �          @�z�@#�
=#�
����i�?n{@#�
?�  ��G��_�A�                                      Bxf�F  �          @��
@7
=>���{��Wff@:=q@7
=?�ff�r�\�L�A�ff                                    Bxf�T�  �          @�(�@4z�=�\)�~{�Z33?�
=@4z�?z�H�vff�P�A���                                    Bxf�cR  �          @�p�@<(�>\�z=q�S(�@�p�@<(�?��
�mp��DA�p�                                    Bxf�q�  �          @�(�@2�\?z��|���X�A;�
@2�\?�p��mp��Fz�A�p�                                    Bxf���  �          @�p�@논��
��33�yp�C���@�?}p�����n�\A���                                    Bxf��D  �          @�@z�>�
=��=q�uffA"�H@z�?�z���33�b  A��                                    Bxf���  �          @�(�@G�>�  ��G��w�
@ƸR@G�?��R����g\)A��H                                    Bxf���  �          @�=q@   >��R��33�i�R@�p�@   ?�G��z=q�Y�A�Q�                                    Bxf��6  �          @��@
=q>�\)�����vQ�@�\)@
=q?�(��vff�d�
A��                                    Bxf���  �          @�(�@�\>������
�}
=A1�@�\?����z=q�g��Bp�                                    Bxf�؂  �          @�  ?��
?Q���G�W
A�G�?��
?޸R�o\)�a\)B1��                                    Bxf��(  �          @��#�
@p���\)��RB�p��#�
@�������{B�B�                                    Bxf���  �          @�p���z�@�33�\���RB��
��z�@��H��ff��=qB��)                                    Bxf�t  �          @��R���
@�Q����иRB�uÿ��
@��Ϳ�{�O�B�Ǯ                                    Bxf�  �          @�����\@��H��
=���B�zῂ�\@�p��Y���!��Ḅ�                                    Bxf�!�  �          @�����@��
����\)Bή���@��8Q����B�                                      Bxf�0f  �          @��R����@��H��ff��{B�����@��Ϳ8Q����B�\)                                    Bxf�?  �          @��R��33@�=q��G�����Bݽq��33@��
�.{��ffB�G�                                    Bxf�M�  �          @�\)����@��H���H��=qB�aH����@�녾\��z�B�G�                                    Bxf�\X  �          @�����33@�
=����G�B����33@��H=�\)?E�B�L�                                    Bxf�j�  �          @��ÿ��\@�  �����Dz�B��ῢ�\@��
=���?���B�G�                                    Bxf�y�  �          @�=q��z�@���W
=�  B��f��z�@�p�>�33@\)BΙ�                                    Bxf��J  �          @�=q�z�H@��(����Q�B�p��z�H@�{?
=q@��
B�\)                                    Bxf���  �          @�=q��{@��H�@  �z�B�LͿ�{@�(�>�(�@�(�B�\                                    Bxf���  �          @�=q��p�@��׿Q����B�aH��p�@��\>�Q�@���B�                                      Bxf��<  �          @��ÿ��H@�\)�Y����B�Q쿺�H@���>��
@h��B��H                                    Bxf���  �          @��ÿǮ@��Ϳ���H��B�{�Ǯ@���=�\)?=p�B�(�                                    Bxf�ш  �          @��ÿ�Q�@��׿=p���RBՊ=��Q�@���>�G�@���B�L�                                    Bxf��.  �          @��ÿ��@��R�\(����B�33���@���>��R@dz�B׽q                                    Bxf���  �          @�G���ff@�\)�O\)���B�ff��ff@���>�Q�@��B�                                      Bxf��z  �          @�33�Ǯ@����5� ��B��Ǯ@��\>�@�z�B���                                    Bxf�   �          @�ff��\@����ff����B�\��\@��\?@  AQ�B�W
                                    Bxf��  �          @���33@��
����Q�B�3�33@��H?=p�A ��B�                                      Bxf�)l  �          @����
@�����p���Q�B�ff��
@��?Q�A��B��H                                    Bxf�8  �          @�
=��p�@�=q��=q�;�B��{��p�@��?n{A#�B�G�                                    Bxf�F�  �          @�Q��  @����33�qG�B�q�  @�p�?Tz�A�B�Q�                                    Bxf�U^  �          @�G����@��R�����c33B�Ǯ���@�z�?W
=A33B�k�                                    Bxf�d  �          @�  �(�@�z��G���  B�ff�(�@���?��A6�\B왚                                    Bxf�r�  �          @�  �p�@�(��k��   B��f�p�@���?n{A"�RB���                                    Bxf��P  �          @�{�   @�G��.{��{B���   @�{?uA+�B�                                    Bxf���  �          @�
=�%�@�G��\)��G�B�Q��%�@�?�  A0  B��                                    Bxf���  �          @��R�\)@�G�>��
@b�\B�k��\)@�=q?�Q�A�G�B��f                                    Bxf��B  �          @����G�@�ff@�A�z�B����G�@s33@E�B=qB�#�                                    Bxf���  �          @��H��\@��@ ��A��\B�(���\@Vff@]p�B!B���                                    Bxf�ʎ  �          @�����@�  @*�HA�G�B��H���@L��@eB)��B�aH                                    Bxf��4  �          @����@��@{A�RBꞸ��@W
=@Z�HB#
=B���                                    Bxf���  �          @�녿��
@���=#�
?
=qB�LͿ��
@��
?�\)A_�
B�                                    Bxf���  �          @�z�
=q@9���X���C��B��
=q@j=q�#33�
(�BĨ�                                    Bxf�&  �          @�z�<��
@7
=�i���O�
B�
=<��
@l(��3�
�33B�B�                                    Bxf��  �          @�33<��
@8���e��L=qB�B�<��
@mp��.�R�z�B�k�                                    Bxf�"r  �          @�{��\)@G��^�R�@�B�\)��\)@y���%��ffB�L�                                    Bxf�1  �          @�
=���H@J�H�\���<=qBų3���H@|���!����B�B�                                    Bxf�?�  �          @�G����@k��7
=�  B�.���@�G���=q���B���                                    Bxf�Nd  �          @�G����R@�����
����B܅���R@�p��u�9B�.                                    Bxf�]
  �          @�=q����@~{���R��G�B�����@�33�fff�-G�B�{                                    Bxf�k�  �          @��H�ٙ�@�33�������B�
=�ٙ�@�ff�E���B���                                    Bxf�zV  �          @��R�k�@��\�(����B�z�k�@��
��  ���Bʔ{                                    Bxf���  �          @�
=�&ff@�(��*�H�(�B��f�&ff@���G���  B��
                                    Bxf���  �          @�Q쾮{@�Q��<����
B�����{@�(�����(�B�\)                                    Bxf��H  �          @��
���H@�(�����=qB��Ὶ�H@��H��33�P(�B�#�                                    Bxf���  �          @��\�.{@�(��7��	��BƔ{�.{@�\)�ٙ����\B�=q                                    Bxf�Ô  �          @�=q�fff@�p��/\)��
B�k��fff@�\)�Ǯ��p�Bɏ\                                    Bxf��:  �          @�=q�@  @���<���G�B��ÿ@  @������p�B�33                                    Bxf���  �          @�=q�n{@�(��1��=qB�\)�n{@��R��{���
B�L�                                    Bxf��  �          @�Q쾅�@����<(��\)B�zᾅ�@��������B�z�                                    Bxf��,  �          @�  �aG�@���:�H��B�G��aG�@���  ���
B�p�                                    Bxf��  �          @�\)��p�@��
�0  ���B�G���p�@�ff�������B�                                    Bxf�x  �          @����@�
=��\)�Y��B�G����@��H?��AF=qB��
                                    Bxf�*  �          @��R�4z�@���>���@x��B��)�4z�@s33?��A��RB�{                                    Bxf�8�  �          @��@��>�33@��\B��3�@�  ?��HA���B�                                    Bxf�Gj  �          @�p��
=q@���>#�
?���B�=q�
=q@�ff?���AyG�B�\)                                    Bxf�V  �          @�=q��Q�@��
>B�\@z�B��ÿ�Q�@��?���A��HB�\                                    Bxf�d�  �          @������@��׿�  ����B�����@��׾��R�s�
B�\)                                    Bxf�s\  �          @��H���@�{������\B��쿧�@�z�\)��G�B�W
                                    Bxf��  �          @����333@}p�?ٙ�A�{B��\�333@W
=@,(�A���C�\                                    Bxf���  �          @����=q@AG�@ffA�{C ���=q@ff@5�B \)C��                                    Bxf��N  �          @����z�?�p�@��\B}�
C33�z��G�@�\)B���C6�R                                    Bxf���  �          @�p���?˅@��
Bj=qC�3��>�z�@�(�B��C,�                                    Bxf���  �          @����'
=?�
=@�(�B_G�CaH�'
=>B�\@�33Bqp�C/��                                    Bxf��@  �          @��R�X��@�
@eB$�
C���X��?�  @��\BA�
C�\                                    Bxf���  �          @�{�O\)@z�@mp�B+��Cp��O\)?�p�@�ffBI�RCE                                    Bxf��  �          @�{�N{@��@r�\B0�C�q�N{?��@��BM�C!ff                                    Bxf��2  �          @�{�L(�@#33@eB%=qCQ��L(�?�p�@�z�BF�HC�                                    Bxf��  �          @�p��P��@\)@j�HB+�\C���P��?�33@���BH��C �\                                    Bxf�~  �          @����P  @  @k�B+�CL��P  ?�z�@���BI=qC aH                                    Bxf�#$  �          @�=q�C33@C�
@Dz�BC�f�C33@ff@r�\B6��Cn                                    Bxf�1�  �          @�\)�8��@e�@(�A�G�C �f�8��@1G�@Tz�B��C(�                                    Bxf�@p  �          @����G
=@`��@�A���C���G
=@-p�@S33B�\C
�R                                    Bxf�O  �          @�=q�0��@_\)@ffAޣ�C ff�0��@-p�@N{B{C��                                    Bxf�]�  �          @�(��0��@P  @3�
BffCT{�0��@ff@fffB3
=C�
                                    Bxf�lb  �          @���.{@Tz�@6ffBp�CL��.{@��@j=qB5
=C
�=                                    Bxf�{  �          @����:�H@aG�@#�
A�(�C���:�H@*�H@\(�B#�C	�{                                    Bxf���  �          @�ff�1G�@�\)@G�A�ffB�p��1G�@`  @G�B�RC Y�                                    Bxf��T  �          @����@���?ٙ�A��\B�����@w�@9��B33B��R                                    Bxf���  �          @�{�;�@c�
@7
=A���Cn�;�@'�@o\)B.z�C
.                                    Bxf���  �          @���*�H@���@(�AָRB�{�*�H@K�@^{B��C�q                                    Bxf��F  �          @�z��$z�@���@(�A�ffB���$z�@K�@^{B!=qC �3                                    Bxf���  �          @�����@�33@-p�A��HB�(����@J�H@p  B1�B�                                    Bxf��  �          @�z��ff@���@   AݮB���ff@QG�@dz�B&�B�\)                                    Bxf��8  �          @��>{@a�@5�A�=qC
=�>{@%@mp�B-33C
��                                    Bxf���  �          @�{��@�ff@p�A���B��f��@dz�@g�B(�B�u�                                    Bxf��  �          @�\)�Z=q@Q�@/\)A�z�C)�Z=q@�@c�
B"G�C33                                    Bxf�*  �          @�(��P��@J=q@9��B=qC�
�P��@p�@k�B,=qC�
                                    Bxf�*�  �          @���W
=@9��@Dz�B
z�C8R�W
=?�z�@qG�B0��Ck�                                    Bxf�9v  �          @���Vff@O\)@.{A�ffC�R�Vff@�@a�B#��C5�                                    Bxf�H  �          @���(��@�(�?�=qAi�B�#��(��@tz�@!�A���B�W
                                    Bxf�V�  �          @�
=��@�?Y��A=qB����@��@	��AÅB�
=                                    Bxf�eh  �          @�{���H@�(�>�p�@�z�B�B����H@�=q?�  A���B�ff                                    Bxf�t  �          @��Ϳ�=q@��\>��?�Q�B���=q@��H?��
A�=qB�Q�                                    Bxf���  �          @�z��,(�@��R?uA-�B���,(�@p��@Q�A��
B�.                                    Bxf��Z  �          @��(�@��>k�@%�B�ff�(�@�z�?���A���B�
=                                    Bxf��   �          @����
=@�p���\)�K�B���
=@���?�=qAB{B�                                    Bxf���  �          @���33@��
��G��3�
B��)�33@�
=>\@��RB�                                    Bxf��L  �          @��R�
�H@�ff��\)�HQ�B噚�
�H@��\?���AC
=B�q                                    Bxf���  �          @�Q��(�@�\)�������B��(�@�{?Y��Ap�B��                                    Bxf�ژ  �          @����=q@��\�^�R�G�B�p��=q@�z�?   @��B��H                                    Bxf��>  �          @����{@��
��{�mp�B�G��{@�Q�?��
A4Q�B�\)                                    Bxf���  �          @������@��H>��?ٙ�B��H����@��\?���A��B��                                    Bxf��  �          @�\)��  @�ff���
�c�
Bծ��  @�Q�?���Aw
=B���                                    Bxf�0  �          @����#�
@�=q����
=B��3�#�
@���=u?:�HB��\                                    Bxf�#�  �          @��R?Q�@���������B�{?Q�@�  �aG��0  B�W
                                    Bxf�2|  �          @���\@�ff�p���I�B�B��\@�G�>�{@��B�
=                                    Bxf�A"  �          @�33���@l(�@�\A��HB�(����@6ff@Q�B/(�B�
=                                    Bxf�O�  �          @�\)�8Q�@333@Tz�B(�C���8Q�?�Q�@�  BH�C��                                    Bxf�^n  �          @���G
=@7�@R�\BffC	@ �G
=?�G�@�  B@Ck�                                    Bxf�m  �          @��
�L��@8Q�@L(�BffC
  �L��?��@z=qB;Q�C��                                    Bxf�{�  �          @�33�Y��@.�R@Dz�B�C=q�Y��?�Q�@p  B2ffC��                                    Bxf��`  �          @�33�4z�@AG�@XQ�B(�C{�4z�?�\)@�(�BJ�C��                                    Bxf��  �          @�ff�k�@�  ?c�
A33B��ÿk�@�  @ffAظRB�33                                    Bxf���  �          @��
>B�\@���\���B���>B�\@��
?��AR�HB��                                     Bxf��R  �          @�Q��=q@�
=?5@�
=B�G���=q@���@
�HA���Bڀ                                     Bxf���  �          @�녿���@�33?^�RA�\B������@��@�\A��HB�.                                    Bxf�Ӟ  �          @���G�@��
?��@��B�
=�G�@�\)@   A���B�\                                    Bxf��D  �          @����ff@�p�?=p�A ��B�u��ff@�\)@�A�p�B�33                                    Bxf���  �          @�
=�'
=@��?n{A#�B�B��'
=@z�H@{A��B�Q�                                    Bxf���  �          @�ff�2�\@���?�  A1�B�Q��2�\@p��@  AͮB��                                    Bxf�6  �          @��'
=@��
?Y��AQ�B�q�'
=@y��@	��Aď\B���                                    Bxf��  �          @��R�Y��@��?�Q�A��\B�\�Y��@��@E�B=qBˣ�                                    Bxf�+�  �          @�
=���@��R?�A��B�𤿅�@x��@QG�BQ�B��                                    Bxf�:(  �          @��R���@�?�(�A��
B�����@g
=@N�RB�RB�#�                                    Bxf�H�  �          @�{�Q�@���?��HA��\B�
=�Q�@]p�@K�B�B�B�                                    Bxf�Wt  �          @�{�9��@~�R?�{A���B�G��9��@R�\@0��B ��CxR                                    Bxf�f  �          @��R�8Q�@���?\A�=qB�\�8Q�@W�@,(�A�{C�=                                    Bxf�t�  �          @�ff�C33@��?��A<  B��)�C33@a�@��A���C�\                                    Bxf��f  �          @���#�
@�=q@�RA��B�G��#�
@K�@X��B�C �{                                    Bxf��  �          @��H�!G�@�33@�\A�B�33�!G�@K�@\��B!�C aH                                    Bxf���  �          @��H���@�?J=qAz�B�����@~{@Q�A���B��H                                    Bxf��X  �          @�(��(�@���   ���B�Ǯ�(�@�
=��  �4��B�                                    Bxf���  �          @�z�:�H@���!���  Bƽq�:�H@�
=���\�7�BĞ�                                    Bxf�̤  �          @����@�(��
�H�ǅB�����@��\�(�����B�W
                                    Bxf��J  �          @���>�z�@n{�`���-
=B�\>�z�@�z��
�H��  B���                                    Bxf���  �          @��O\)@��\��
��33B�{�O\)@����\���B�k�                                    Bxf���  �          @�{�Q�@��ÿ����BǊ=�Q�@���W
=�33B�L�                                    Bxf�<  �          @�z῀  @��H��(��]p�B�z῀  @�
=>�p�@�
=B��
                                    Bxf��  �          @��?˅@����(��bBXz�?˅@X���N{�"�RB�Ǯ                                    Bxf�$�  �          @��\@%�?L�������g�\A�z�@%�@Q��u��CffB�                                    Bxf�3.  �          @��
@(�?(����{�q��AqG�@(�@�\��  �NQ�Bz�                                    Bxf�A�  �          @��
?�?�����\)�u�RB#�?�@<���o\)�<ffBg�                                    Bxf�Pz  �          @�?��R?˅��ffW
B;=q?��R@@  �|���F��B~�\                                    Bxf�_   �          @�?n{@z����
�\)B��
?n{@\(��n{�8�RB��H                                    Bxf�m�  �          @��?�Q�?����33�h33B0��?�Q�@K��`���-{Bjff                                    Bxf�|l  �          @�ff@(�?�������w
=A�{@(�@ ���}p��G�BC�R                                    Bxf��  �          @���@  @k��AG��{Bj  @  @��R��
=���\B|��                                    Bxf���  �          @���@@dz��S33�(�Bn��@@�ff��(�����B��                                    Bxf��^  �          @��@  @w��5�p�Bo33@  @��\��
=�{
=B=q                                    Bxf��  �          @�=q?���@xQ��H���B��R?���@�{���H��=qB��=                                    Bxf�Ū  �          @�G�?�\)@xQ��L(���
B���?�\)@��R��G����
B�                                      Bxf��P  �          @���?��@����;��{B��?��@�(����yB�L�                                    Bxf���  �          @���?��@~�R�B�\��HB�G�?��@�  ��=q��33B���                                    Bxf��  �          @�Q�?�\@p���L���p�B��{?�\@�33��ff��ffB��f                                    Bxf� B  �          @�
=?�\@e��W
=� G�B~�?�\@�\)�   ���\B��H                                    Bxf��  �          @�
=?�p�@|(��C�
�  B��)?�p�@�\)������
=B�=q                                    Bxf��  �          @�?xQ�@����;��
  B���?xQ�@�(���33�{�
B��                                     Bxf�,4  �          @�{?�{@�(��;��	B��H?�{@����z��|z�B�B�                                    Bxf�:�  �          @��?�z�@xQ��S33��B���?�z�@�  ��=q��z�B���                                    Bxf�I�  �          @���?�Q�@�=q�.�R��
=B�{?�Q�@�\)��33�J�HB�                                    Bxf�X&  �          @�Q�?���@��H�-p���
=B��?���@�  ��\)�E�B��{                                    Bxf�f�  �          @�  ?xQ�@{��Tz��p�B�B�?xQ�@�녿�=q���B�33                                    Bxf�ur  �          @�ff?�  @A��w��C\)B~�\?�  @�z��(����p�B�(�                                    Bxf��  �          @�33�,(�@aG�@5B�\B��q�,(�@Q�@vffB<
=C
�\                                    Bxf���  �          @���X��@\��@��A�p�C���X��@��@Z�HB  C0�                                    Bxf��d  �          @���I��@Q�@:=qB��C޸�I��@
=@uB5p�C#�                                    Bxf��
  �          @��Ϳ���@�  ����z�B�p�����@�����\)�=p�B�33                                    Bxf���  �          @��=p�@�ff�h���$Q�B�\�=p�@�
=?O\)A{B�                                      Bxf��V  �          @�=q��(�@�\)����B�L;�(�@��\?��Af�\B��\                                    Bxf���  �          @��?�@�\)��Q���p�B���?�@�Q�=�Q�?k�B���                                    Bxf��  �          @�����@�  �   ��(�B��ͼ�@�33�k��,z�B��3                                    Bxf��H  �          @�z�#�
@p  �Fff�ffB��;#�
@�=q��33���RB��                                    Bxf��  �          @�33?�\)?�p�����|�Bqz�?�\)@XQ��c33�3��B�8R                                    Bxf��  �          @�
=?�(�?��H��p��~�A�z�?�(�@.{�~�R�GQ�BX�\                                    Bxf�%:  �          @�@G�?�����m{A�Q�@G�@1G��mp��8(�BJ��                                    Bxf�3�  �          @���@"�\@z��_\)�;z�B��@"�\@HQ��%��ffBK��                                    Bxf�B�  �          @��@!G�?L������l33A��R@!G�@���w��C\)B'ff                                    Bxf�Q,  T          @�
=@�?˅���H�\ffB33@�@8���S33�%�BJ��                                    Bxf�_�  T          @�Q�@;�?z�H��(��TA�ff@;�@�b�\�-z�Bp�                                    Bxf�nx  �          @��H@%?�=q��p��f=qA�33@%@!��qG��8B1p�                                    Bxf�}  �          @��?��H?n{����
A�{?��H@"�\����Sz�BQQ�                                    Bxf���  �          @���@'�?�����Q��c33Aڏ\@'�@5��p���1G�B<��                                    Bxf��j  �          @�?�Q�@B�\���
�GQ�Bt
=?�Q�@����3�
��B�                                      Bxf��  �          @�{?�  @E��
=�K=qB��?�  @���8Q�� �B��                                    Bxf���  �          @�p�?�(�@5���G��QG�Bj�?�(�@����B�\�z�B��                                    Bxf��\  �          @�z�?�=q@.�R����Xz�Bp{?�=q@��\�H����B��q                                    Bxf��  �          @���?��@'
=�����e�B}
=?��@����W
=�ffB��)                                    Bxf��  �          @�{?�(�@7
=��(��W�B{33?�(�@��R�G
=�B��=                                    Bxf��N  �          @�\)?�\)@5���G��N�\Bb�\?�\)@����A��ffB���                                    Bxf� �  �          @�  @Q�@"�\��(��Sz�BHz�@Q�@z�H�Mp��{Bv
=                                    Bxf��  �          @���@Q�@{���R�WffB+�
@Q�@h���X�����Bc33                                    Bxf�@  U          @�  @1�?�(���  �\=qA�{@1�@<���l���)ffB:�
                                    Bxf�,�  �          @�  @@��?z�H��  �[��A�ff@@��@   �w��2�B�                                    Bxf�;�  �          @�  @B�\?�������V�
A�
=@B�\@$z��o\)�,��B �                                    Bxf�J2  �          @���@33@5������J�BX\)@33@���?\)�G�B                                      Bxf�X�  �          @��@�@'
=��z��R33BL{@�@�  �K��\)Bx�                                    Bxf�g~  �          @��@
=q@{��p��b  B7
=@
=q@n�R�e���Bo��                                    Bxf�v$  �          @�33@  @  ��33�]p�B3��@  @o\)�`  �
=Bk�H                                    Bxf���  �          @�z�@��@  ��=q�X��B,ff@��@n�R�^{�{Bd��                                    Bxf��p  �          @��
@��@#33��z��M�B;�@��@}p��L(��
��Bkp�                                    Bxf��  �          @��H@��@8Q������H  BU�@��@�
=�>{� =qB|ff                                    Bxf���  �          @��H@?��
��\)�h��B�@@W
=�q��+�B\��                                    Bxf��b  �          @��@@�R��=q�[
=B.��@@n{�^{�{Bg�                                    Bxf��  �          @��@��?����ff�d(�Bz�@��@\���mp��%�
B\��                                    Bxf�ܮ  U          @��@��@p������Yp�B*G�@��@l���\���(�Bc��                                    Bxf��T  �          @��@"�\?�33��33�]�RB\)@"�\@[��g
=� �RBV�                                    Bxf���  �          @��
@z�?�\)��33�n�HBG�@z�@P���|(��2
=BZ
=                                    Bxf��  �          @��@�?�{��G��g=qB33@�@]p��r�\�(33B^\)                                    Bxf�F  �          @��?�ff@{�����e=qBW�\?�ff@����b�\�  B���                                    Bxf�%�  �          @�ff?W
=@9������gffB��\?W
=@�ff�Z�H�Q�B��                                    Bxf�4�  �          @���?�ff@)�����R�m(�Bff?�ff@����j=q�Q�B�(�                                    Bxf�C8  �          @���?���@AG���Q��_��B���?���@����U����B��                                    Bxf�Q�  �          @��H>�\)@l����Q��J33B�B�>�\)@��H�6ff��G�B���                                    Bxf�`�  �          @��
>�Q�@i�����H�M�\B���>�Q�@�=q�;����B���                                    Bxf�o*  T          @����#�
@{���\)�<p�B��=�#�
@��R�   ����B�p�                                    Bxf�}�  T          @�ff�L��@���xQ��,p�B�ff�L��@�ff�
=���HBŔ{                                    Bxf��v  �          @��
��\)@��c�
��B�{��\)@�ff��\��G�B�aH                                    Bxf��  �          @�z��  @�ff�G
=�(�B�Q��  @��ÿ�  �M��Bӽq                                    Bxf���  �          @��׿���@�\)�R�\�
��Bڔ{����@�(���33�a�BԀ                                     Bxf��h  �          @��R��@�z��p  �$ffBӨ���@�\)������(�B�G�                                    Bxf��  �          @�  ��\)@�ff�c�
�G�B�B���\)@�
=��  ���B֏\                                    Bxf�մ  �          @�녿�ff@���Tz��ffB�W
��ff@��\��Q��f{B�G�                                    Bxf��Z  �          @��׿�ff@�(��Y����
B�  ��ff@��\���
�v=qB�p�                                    Bxf��   �          @�녿�{@�
=�`���=qB��῎{@�ff�˅�33B�\                                    Bxf��  �          @�{��z�@���Mp����B����z�@�G���=q�U�B۳3                                    Bxf�L  �          @�����@g��
=q��(�C�H���@c�
?\(�A�C)                                    Bxf��  T          @�Q���p�@(�?G�@�ffC�=��p�?�(�?У�A}�C 
=                                    Bxf�-�  S          @������@Vff>�\)@,(�C�����@@  ?\Al  C�)                                    Bxf�<>  U          @�\)��33@^{�����Cs3��33@N{?��
AF{Ch�                                    Bxf�J�  �          @�����G�@QG�>�
=@���C  ��G�@7�?�\)Az{CQ�                                    Bxf�Y�  �          @�=q���
@P  >8Q�?�Q�C�����
@<(�?��AT(�C�                                    Bxf�h0  �          @�
=���@Q�=�Q�?^�RC�����@@  ?���AL��C�R                                    Bxf�v�  �          @�������@R�\<��
>B�\C�
����@B�\?�G�AA�C��                                    Bxf��|  �          @�=q��  @Z�H<�>�z�C����  @I��?�=qAJ=qC                                    Bxf��"  �          @�33���\@U?��@�(�C����\@7
=?���A��HC��                                    Bxf���  �          @�G�����@j�H<#�
=���C�����@X��?�33AW�C��                                    Bxf��n  �          @�����33@X��?\(�A�C���33@3�
@�A��C�                                    Bxf��  �          @�Q����@P  �#�
����CE���@@��?�p�A<��C@                                     Bxf�κ  �          @�  ����@Z=q>L��?��HC!H����@E�?��RAg�C޸                                    Bxf��`  �          @�Q��n�R@p  �W
=��C�=�n�R@r�\?(��@ᙚC�                                     Bxf��  
�          @�ff�Q�@�Q��#33��(�B�\�Q�@�z��R�ǮB垸                                    Bxf���  �          @���!�@����(���(�B�.�!�@�(��u�=qB�                                    Bxf�	R  �          @�33�p�@��
��R���B�=q�p�@��
��\)�5B���                                    Bxf��  �          @��\���@��
�J=q�
�B�B����@�  ��p��K�Bѣ�                                    Bxf�&�  �          @�
=��=q@��\�e��  B�\��=q@��
�У����\B�                                    Bxf�5D  �          @���(�@��
�S�
��B����(�@�녿�{�]p�B��f                                    Bxf�C�  �          @��\@����W
=���B��ÿ\@�Q쿷
=�j�\B�8R                                    Bxf�R�  �          @���n{@�
=�hQ�� G�B��Ϳn{@������H��33B��)                                    Bxf�a6  �          @���!G�@����P  ��
B�Ǯ�!G�@�p����R�L(�B�(�                                    Bxf�o�  �          @��H=u@���s�
�,��B�.=u@�
=��
=��Q�B��{                                    Bxf�~�  �          @��Ϳh��@z�H�{��1G�B�
=�h��@������z�B�                                      Bxf��(  �          @��G�@�=q�e��BȀ �G�@�(���\)���\Bą                                    Bxf���  �          @�녿   @��\�\����B�33�   @��\��p��w�B��R                                    Bxf��t  T          @�����@����j=q�!p�B\���@����Q���p�B���                                    Bxf��  �          @�{���
@}p������5B�k����
@�
=�������B��
                                    Bxf���  �          @�33���@hQ�����M�B�=q���@�(��1G���z�B���                                    Bxf��f  �          @�33���@n{��  �I��B�����@�{�*�H���
B�\)                                    Bxf��  �          @�\)�e@��
�0���޸RCp��e@�G�?��A0(�C�R                                    Bxf��  �          @�\)�mp�@��\�����\C���mp�@���?���A��
C��                                    Bxf�X  �          @�{���H@p  ?p��A�\C	u����H@Dz�@ffAř�C{                                    Bxf��  �          @�\)�l��@�p�?���A_33C�f�l��@Q�@:�HA�\C
�                                    Bxf��  �          @�Q��aG�@���?
=@��
B�Ǯ�aG�@z=q@�A�\)C�                                    Bxf�.J  �          @����s33@�ff?z�HAC(��s33@]p�@$z�Aԣ�C	�                                    Bxf�<�  �          @���vff@J�H@,(�A�RC���vff?�@n�RB$��C��                                    Bxf�K�  �          @�  �z=q@S�
@%�A�C���z=q@ ��@l(�B��C�=                                    Bxf�Z<  �          @��H��p�@�z��{��-�RB�B���p�@���������B��                                    Bxf�h�  S          @�=q>\)@�(����\�2z�B�  >\)@�p����z�B���                                    Bxf�w�  U          @���=#�
@�(��n�R�!�RB��
=#�
@�Q��z���z�B�{                                    Bxf��.  �          @��>\)@|(��qG��.�
B��f>\)@��
��\)��z�B��
                                    Bxf���  �          @���?�
=@����  �\��BN�
?�
=@����HQ���RB���                                    Bxf��z  �          @���@(�?�{����m��B!=q@(�@j=q�l���#�
BlQ�                                    Bxf��   �          @��@%?�
=��p��gQ�B��@%@aG��tz��$�BV�                                    Bxf���  �          @���@-p�?�����
=�\z�B	\)@-p�@fff�dz��Q�BS��                                    Bxf��l  �          @��H@C33@z������KQ�Bz�@C33@n{�S33�	�
BJ�\                                    Bxf��  �          @���@QG�?�{���H�Q=qA��\@QG�@Fff�h���B-�
                                    Bxf��  �          @�Q�@5�?�  ��(��iffA��@5�@8�������4�B633                                    Bxf��^  �          @��R@/\)@#33��  �Bz�B+��@/\)@���6ff��B_�R                                    Bxf�
  �          @�  @#33?�������u33A��@#33@@  ��p��:�RBFp�                                    Bxf��  �          @��@{?���G��v{A��@{@G���(��8
=BN�                                    Bxf�'P  �          @�  @'
=?fff����n�HA���@'
=@0���}p��8��B:(�                                    Bxf�5�  �          @�
=@8Q�?Tz���G��b{A�
@8Q�@'
=�s33�1p�B(z�                                    Bxf�D�  �          @�  @-p�?����Q��]ffA�  @-p�@P  �_\)�  BHp�                                    Bxf�SB  �          @�  @#�
@���=q�R�B=q@#�
@j�H�E�
�B\=q                                    Bxf�a�  �          @�\)@�\@>�R��33�B��B^�R@�\@����!G���p�B�aH                                    Bxf�p�  �          @��?��@=p���z��J�
Bs�?��@����#�
��z�B�\                                    Bxf�4  �          @��?�(�@4z����R�Z{Bz(�?�(�@���:=q� {B�33                                    Bxf���  �          @���@G�@:=q��  �H�\B\��@G�@����*�H��RB��                                    Bxf���  �          @�G�@�H@@  �{��6�BL�R@�H@�33���
=Bs��                                    Bxf��&  �          @��
@#33@;���=q�9=qBC�
@#33@�33�\)��ffBnff                                    Bxf���  �          @�@)��@B�\�\)�2�
BC��@)��@�p��Q���  Bl(�                                    Bxf��r  �          @�@��@P������:�Bc{@��@�p�����ǮB��)                                    Bxf��  �          @�(�?�G�@XQ����
�<Bz
=?�G�@�G����G�B���                                    Bxf��  �          @�(�?�(�@X����(��=ffB|G�?�(�@����ff�ř�B��                                    Bxf��d  �          @���?�\)@%����b  BhG�?�\)@����G��	�B���                                    Bxf�
  �          @���?��@G���  �y=qB���?��@�G��XQ���
B�.                                    Bxf��  �          @���?�z�@Y���c�
�-{B�
?�z�@�=q�������
B��                                    Bxf� V  �          @��@&ff@&ff�h���3{B3��@&ff@xQ��{�ˮB`�
                                    Bxf�.�  �          @�33@{@c33�U����Bh  @{@��
�������B��                                    Bxf�=�  �          @�(�?�@x���G
=�z�B~��?�@��\��Q��N�\B��\                                    Bxf�LH  �          @��?�@n{�QG��\)B�8R?�@�  ��z��w�B�aH                                    Bxf�Z�  �          @�  @@N{�AG����BW��@@�{�����G�Bs\)                                    Bxf�i�  �          @�\)@ ��@.�R�;��ffB=�@ ��@mp���G���33B_��                                    Bxf�x:  �          @�Q�@\��@��R�\���B ��@\��@W
=�z����
B1                                      Bxf���  �          @��H@���?(��G����A�R@���?�\)�$z���
=A���                                    Bxf���  �          @�\)@���?0���U��RA�@���@�\�.�R��(�A�Q�                                    Bxf��,  
�          @��@�=q>k��U��R@?\)@�=q?�\)�;��A�Q�                                    Bxf���  �          @�{@���   �G
=�p�C��H@��?^�R�A��	�A4��                                    Bxf��x  �          @�33@w
=�Y���L(����C�@w
=?z��P  �\)A	p�                                    Bxf��  �          @��R@w���\)�E�p�C���@w�����^�R�'�
C��                                     Bxf���  �          @�G�@w
=�����Fff��C�^�@w
=�����e��*�C��\                                    Bxf��j  �          @��
@mp������G���C�N@mp����`���-z�C�                                    Bxf��  �          @��@`�׿�=q�L(����C�� @`�׽�\)�c�
�5��C�q�                                    Bxf�
�  �          @�z�@^�R� ���L(����C��@^�R��ff�o\)�;z�C�Q�                                    Bxf�\  �          @�z�@G
=��33�g��3G�C�N@G
=�.{���\�R�RC�xR                                    Bxf�(  �          @�z�@Z�H�˅�G��\)C�� @Z�H�����`  �6�C��                                    Bxf�6�  �          @��@<�Ϳ�{�\���6�C���@<��<��
�s�
�P�
>��                                    Bxf�EN  
�          @�=q@��z���  �Pp�C�'�@������Q��z�\C�                                      Bxf�S�  �          @���?��ÿ��H��ff�f{C�u�?���=#�
��z��?���                                    Bxf�b�  �          @��?���\)�xQ��dQ�C���?�>W
=��ff�@���                                    Bxf�q@  �          @�  ��p��\(���G��Cv����p�?�
=��
=B��H                                    Bxf��  �          @������=��
����\C0�
����?�(��|(��k�RB���                                    Bxf���  �          @�(��&ff?�{�h���H  C=q�&ff@AG��%�  C��                                    Bxf��2  �          @�p��U�?�(��N{�&{CB��U�@.{�G�����C�{                                    Bxf���  �          @�G��\��?�
=�K���HC�\��@8������ˮC�q                                    Bxf��~  �          @�33�
�H?u��=q�o=qC
=�
�H@(Q��P  �.��Ck�                                    Bxf��$  �          @��R��{?�=q��\)�w��C����{@P���G��"�B�{                                    Bxf���  �          @�\)�   >�  �����3C,�   @ff�p���Qp�C�f                                    Bxf��p  �          @�zΌ��?B�\��{z�CaH����@'��j�H�K�B�
=                                    Bxf��  �          @��\���
?޸R�\)�o\)CY����
@S33�5��p�B�R                                    Bxf��  �          @��?:�H?�G��Z�H�qBo��?:�H@(Q��!G��+{B��H                                    Bxf�b  �          @p��?����
�=q�2ffC���?�����<���l�\C��=                                    Bxf�!  �          @P��?��H�޸R?!G�A�ffC�e?��H���L�����C��f                                    Bxf�/�  �          @b�\?z�H?@  �N{B��?z�H@��#�
�@z�B�\)                                    Bxf�>T  �          @]p�?
=q?k��N{.Bm�
?
=q@\)�\)�<�B��                                    Bxf�L�  �          @J�H?+�>��R�Dz��
A�ff?+�?У��'��`ffB�\                                    Bxf�[�  �          @�ff?�녾W
=������C��?��?ٙ��{��z=qB`�H                                    Bxf�jF  �          @�p�?aG��^�R�}p�(�C���?aG�?u�|(�k�B=Q�                                    Bxf�x�  �          @|(�?Tzῴz��c�
�=C�K�?Tz�>����tz��
A�33                                    Bxf���  �          @vff?Tz�0���i��8RC��?Tz�?}p��e�Q�BI33                                    Bxf��8  �          @AG�?J=q=�Q��+�@�
=?J=q?�G��
=�g(�Bh(�                                    Bxf���  �          @l��?��?���Q��=��B3�?��@(��������
Bf�\                                    Bxf���  �          @�R?�  ?s33��G��>�\B-(�?�  ?\�p����{Bb{                                    Bxf��*  	a          ?�(�?W
=?Ǯ�c�
���Bv��?W
=?����G��B�\B�Ǯ                                    Bxf���  
(          @&ff?�{?˅�\�  BF
=?�{@ff�(���k33Bdff                                    Bxf��v  �          @
=?�
=?E��Ǯ�?z�B��?�
=?��׿�����BE�                                    Bxf��  �          @e@�\?�z��G�� ��A��@�\@G���z�����B2��                                    Bxf���  T          @q�@{?����&ff�4�A���@{@�޸R��\)B9                                    Bxf�h  
�          @�  @#33?�Q��fff�F��B33@#33@G��{��(�BJ��                                    Bxf�  �          @��>�z�@L�Ϳ�  �ȏ\B��R>�z�@a�=��
?���B��                                    Bxf�(�  "          @����&ff@�ff>Ǯ@�{B����&ff@���@��Aأ�B��q                                    Bxf�7Z  
�          @���U@�z�?Tz�A	p�B����U@c33@-p�A�Q�CB�                                    Bxf�F   T          @����xQ�@|��?��
A(��Cs3�xQ�@Dz�@,(�A�ffC��                                    Bxf�T�  �          @������@[�?}p�A�C  ����@'�@�HA�=qC\                                    Bxf�cL  �          @�(���ff@k�?}p�A  Cu���ff@5@"�\A�G�Cz�                                    Bxf�q�  
�          @�p���=q@_\)?���AS\)C����=q@ ��@1G�A�(�C.                                    Bxf���  
�          @�
=��{@��H?���A)C����{@J=q@4z�A��C                                      Bxf��>  
�          @\�|(�@�=q?+�@ʏ\C�q�|(�@q�@*=qAϙ�C+�                                    Bxf���  �          @��\�Z=q@���>�p�@dz�B��R�Z=q@��
@\)A��C�{                                    Bxf���  "          @�ff�R�\@��
<#�
=��
B��f�R�\@��
@A�33C ��                                    Bxf��0  T          @�=q�C33@�(������|��B����C33@�G�?���A��B���                                    Bxf���  
�          @����C�
@�ff�u�!B���C�
@���?�33AB{B���                                    Bxf��|  
�          @��H@!��2�\�*=q�33C��@!녿��
�h���P\)C��3                                    Bxf��"  
�          @�G�@A��G��QG��#=qC��@A녿
=q�|���P33C��                                    Bxf���  "          @�Q�@@  ����c33�4=qC��3@@  ��\)�����T��C�\)                                    Bxf�n  
�          @���@7���p��[��6�RC�u�@7�<��
�u�T��>�{                                    Bxf�  
�          @��\@^{��A���C�s3@^{���H�i���8�C�                                      Bxf�!�  
Z          @�  @j�H����=p��	��C�9�@j�H�(���j=q�1�RC���                                    Bxf�0`  
(          @�p�@(��W
=����t=qC�s3@(�?�
=����l�\A���                                    Bxf�?  
�          @�Q��   ?��H�fff�]�
C���   @<(��!���
B�G�                                    Bxf�M�  T          @�ff���@1��$z��  C�����@h�ÿ��\�R�\B���                                    Bxf�\R  T          @��R���@W
=�c�
�
=C
���@X��?=p�@�z�C��                                    Bxf�j�  �          @����(�@�\=�?�
=C\)��(�?�p�?�z�AX��C�H                                    Bxf�y�  �          @�(��]p�@�
@G�A�\)C#��]p�?L��@=p�B�\C'�                                    Bxf��D  �          @����J�H?�ff@��A�G�Cs3�J�H?
=@1�B#33C)�                                     Bxf���  �          @~�R�XQ�?#�
?��A噚C)Y��XQ쾙��?�(�A��\C9\                                    Bxf���  �          @����{�>���>.{@\)C/���{�>B�\>�z�@�ffC1L�                                    Bxf��6  
Z          @�33�dz�.{�����(�C>Ǯ�dz�>aG���z���ffC0u�                                    Bxf���  �          @�(��%����;��A�RC9���%?�\)�.{�/�C��                                    Bxf�т  "          @��\�C�
��Q���H�
CP�)�C�
�\�;��-�HC;#�                                    Bxf��(  "          @����k����������{CO�H�k��fff�\)�
=CA�=                                    Bxf���  
�          @����p�������
=��CKǮ��p������
�H��CA��                                    Bxf��t  "          @�33���H��
=>��R@N{CB�3���H��z�\�~{CB�                                    Bxf�  
�          @�Q���녾��?�\)AR{C7\��녿@  ?aG�A#�C<�
                                    Bxf��  �          @8���?�?\BG�C$�{��L��?�{B�C9�                                     Bxf�)f  
�          ?�\)���ͽ�G�?z�HB�ffCB�=���Ϳ��?Q�BI�
Ci�3                                    Bxf�8  �          @'����?��R?�\)B%=qB�B����?.{?�z�B��B�p�                                    Bxf�F�  
�          ?W
=>�G�?�R����(�BZ>�G�?\)>��A��HBO�H                                    Bxf�UX  �          ?��
<�=��
���«�)B�W
<�?���z�H�i�
B��)                                    Bxf�c�  �          @333��ff?�\)�\�p�B��)��ff@
=���(  B��                                    Bxf�r�  T          @Z�H�@��?�녿�z���Q�CJ=�@��?�ff�   �G�C�\                                    Bxf��J  �          @fff�W
=?��z�H���HC+B��W
=?h�ÿ!G��#�C$Ǯ                                    Bxf���  �          @@��� �׿\(�>.{@p��CF��� �׿Y����  ���HCF�
                                    Bxf���  �          @AG��%?\(�?�A,��C!� �%?�\?\(�A�{C(��                                    Bxf��<  
�          @>�R�
=>�z�?:�HA�Q�C,��
=���
?G�A��C6.                                    Bxf���  
�          @   ?c�
?���8Q���=qBp�?c�
?ٙ�=��
@!�Bz
=                                    Bxf�ʈ  �          @.{>W
=?�p��Ǯ�h��B�Ǯ>W
=?��R>�33AQG�B��H                                    Bxf��.  �          @~�R�a녿�(�?��A���CG{�a녿�
=?�A�CM�                                    Bxf���  T          @�{�h�ÿ��\?���A�ffCG5��h�ÿ�  ?��A�RCM�R                                    Bxf��z  T          @�
=����?aG�?�A�G�C*�\���׽�\)@ ��A��\C4Ǯ                                    Bxf�   
�          @��
���?У�@�A�ffC"@ ���>�p�@1�A�{C/�{                                    Bxf��  T          @����?��@(�A�ffC%�
���>��@#�
A�  C2W
                                    Bxf�"l  �          @�{��녾���@   A�C7ff��녿�p�?�{A|  C@z�                                    Bxf�1  �          @�����?0��@#�
A�=qC,�H��녿��@%�A�{C:k�                                    Bxf�?�  �          @��
����?�\)@�RA���C&�����>#�
@'
=A���C2c�                                    Bxf�N^  �          @�Q����@�@
=qA�Q�C  ���?h��@;�A�
=C*p�                                    Bxf�]  T          @��
����@9��@�
A�p�C+�����?˅@L(�A�RC#E                                    Bxf�k�  T          @ȣ����@>{?�\A�33Ch����?��@>{AᙚC!.                                    Bxf�zP  �          @љ��\�����@�RA�=qCkY��\����=q�.{��(�Cn@                                     Bxf���  T          @��H�\)��@
=A�=qCe��\)���������Ch��                                    Bxf���  
�          @�
=�o\)��Q�@G�A�=qCg���o\)��Q쾅����Cj��                                    Bxf��B  T          @�  �r�\����@
�HA�=qCg���r�\��  ��Q��L��Cj+�                                    Bxf���  
Z          @�\)�n{��ff@��A�{Cg���n{������\)�
=Cj�                                    Bxf�Î  �          @�p��j�H���R@�A��RCh{�j�H��  �8Q��\)Ck\                                    Bxf��4  �          @�z��j�H��
=@\)A�Q�Ch
�j�H���R��=q�=qCj�
                                    Bxf���  
�          @�z��p�����@��A��Cg��p�����;�\)�"�\Ciٚ                                    Bxf��  �          @�p���  ���\@G�A��RCd�H��  �������hQ�CgG�                                    Bxf��&  
�          @�Q���  ����@��A�  CeG���  ��������:=qCg�                                    Bxf��  �          @θR�b�\����@(�A��\Cip��b�\�������33Cl��                                    Bxf�r  
Z          @�p��\(����@Q�A�
=CjaH�\(�����L�Ϳ�=qCmE                                    Bxf�*  T          @��fff��(�@ffA�
=Ci���fff��G�������Ck�                                     Bxf�8�  �          @�p��p  ���
?���A�(�ChxR�p  ���Ϳ:�H��  Cj                                      Bxf�Gd  T          @�
=�w
=���\?�33A��HCgaH�w
=����#�
��z�Ci+�                                    Bxf�V
  �          @�\)�b�\��Q�@�A�ffCj�q�b�\���
�(���{Cl�
                                    Bxf�d�  "          @����K���  ?��
AY�Co�q�K���녿���5p�CpG�                                    Bxf�sV  
�          @�G��W
=��p�?�Q�AJ�HCn8R�W
=��ff����<��CnW
                                    Bxf���  	�          @�p��`����?У�AbffCm+��`���������"{Cm�R                                    Bxf���  �          @�  �G
=��
=?�AZ�\Cm��G
=��G������-�CnQ�                                    Bxf��H  �          @�\)�#33���?�G�Atz�Cr�)�#33��G����
�%Cs.                                    Bxf���  �          @�Q��1���33?\Aj�\Cq\)�1���ff��{�+33Cq�
                                    Bxf���  �          @�  �<(���Q�?�\)Ap��Cp�<(���zῌ���#33Cq\)                                    Bxf��:  T          @�ff�0����=q?�  A`��Cr���0����zῠ  �:{Cr�                                    Bxf���  �          @��
� ������?�=qAe�Cv�� �����R��=q�@z�CvB�                                    Bxf��  �          @Ǯ�
�H��z�?�p�A\(�Cx�3�
�H������S
=Cy                                      Bxf��,  �          @�G��{���?���Alz�Cv\�{��zῢ�\�:�RCvaH                                    Bxf��  �          @����{��(�?�z�Au��Cxz��{�����G��9G�Cx�{                                    Bxf�x  �          @���������?�G�A�(�Cy� ����G���Q��.ffCz8R                                    Bxf�#  �          @�Q��/\)��p�?��
A��Co���/\)��ff�333��Q�Cq�                                    Bxf�1�  �          @�����H?:�H=�Q�?���C)����H?
=>�(�@���C+�                                     Bxf�@j  �          @�p��J=q@��;L�Ϳ�B��{�J=q@�z�@��A��B��                                     Bxf�O  �          @�\)�N{@�33���H��ffB����N{@�Q�?�ffA�
=B�#�                                    Bxf�]�  �          @����g
=�#�
@�\A�ffCWW
�g
=�Vff?B�\AQ�C^�{                                    Bxf�l\  �          @�(���{�H��@5�A��CU����{���
?uA33C]L�                                    Bxf�{  �          @�
=��G��8Q�@0  A���CQǮ��G��vff?�G�A��CYn                                    Bxf���  
(          @�G���p��Q�@
=A��\CH����p��@��?��
AffCO��                                    Bxf��N  �          @����(��J�H?��RA��CRxR��(��o\)>L��?��CV                                    Bxf���  �          @Ӆ����Q�@�HA�  CT�3�����G�?�@�33CZ\)                                    Bxf���  �          @�ff�|�����H?��RA��Ce@ �|����\)���H���Cg�=                                    Bxf��@  �          @��H���\��{?��HA��Cc^����\���\��ff���Ce�                                    Bxf���  �          @θR����k�@��A�
=CY�������=q>8Q�?��C^L�                                    Bxf��  �          @�G���G��\(�@33A�33CVG���G����
>�{@<��C[B�                                    Bxf��2  �          @����
=�e�@�A�z�CY.��
=���>\@Y��C^ff                                    Bxf���  �          @���G��K�@�
A�G�CU�
��G��y��>��@��HC[0�                                    Bxf�~  �          @�p���{�L��@�A��CVE��{�|��?�\@��C\�                                    Bxf�$  �          @��
���E�@(�A�=qCUL����xQ�?!G�@���C[�)                                    Bxf�*�  �          @�(���  �K�@*=qA�Q�CW@ ��  ���\?E�@�C^(�                                    Bxf�9p  �          @����Q��3�
@@  A㙚CR����Q��z�H?��RA6{C[n                                    Bxf�H  T          @ə���{�Q�@Dz�A�(�CM���{�e�?\A_�
CW��                                    Bxf�V�  �          @ə���G���\@`��B�HCK���G��`  @�
A�G�CX#�                                    Bxf�eb  �          @�=q������@UA���CN:������l��?�  A��CY�
                                    Bxf�t  �          @Ǯ�����
=q@N�RA��HCK������^{?�G�A��HCWB�                                    Bxf���  �          @�����G��%�@0��A�33CM�R��G��fff?�33A"ffCV:�                                    Bxf��T  �          @�{�����9��@G�A�p�CP�q�����h��?\)@�Q�CV��                                    Bxf���  �          @�  �����9��@
�HA��
CP@ �����e>��@��
CU�H                                    Bxf���  T          @�����G��1�@�RA�{CN����G��`��?z�@�G�CTY�                                    Bxf��F  T          @�\)��=q�,��@'�A���CM����=q�h��?uA�CU�                                    Bxf���  �          @ָR���H�@��@,��A�ffCQ\)���H�|��?aG�@�G�CXs3                                    Bxf�ڒ  �          @���=q�*=q@A�A�=qCN�)��=q�s�
?�=qA8Q�CW�H                                    Bxf��8  
�          @�
=���Ϳ\@4z�AƸRCBk������-p�?��HAl  CL�f                                    Bxf���  
(          @�����  �У�@9��A���CC�
��  �5?�(�Ap(�CNT{                                    Bxf��  T          @Ϯ���\�33@$z�A��HCH#����\�C33?�p�A.�HCP��                                    Bxf�*  �          @�(����R�9��@=qA�(�CO�R���R�mp�?+�@���CV.                                    Bxf�#�  �          @��H��ff��Q�@33A��CF����ff�4z�?�ffA�CNT{                                    Bxf�2v  T          @���ff?��@	��A��\C*���ff���@Q�A�Q�C5}q                                    Bxf�A  �          @�  ���\?��R?��RA�p�C!����\?@  @.{A�z�C,�f                                    Bxf�O�  �          @Ϯ��(�@z�?�G�A{�
C ����(�?n{@#33A�\)C+                                      Bxf�^h  �          @�{��?���?��RA��C&�H��>aG�@�HA�G�C1��                                    Bxf�m  �          @�ff��
=?O\)@G�A�G�C,T{��
=��
=@�A���C8
=                                    Bxf�{�  �          @�����\)?G�@��A�(�C,�{��\)�\@\)A�=qC7��                                    Bxf��Z  �          @�(����>Ǯ@ffA�
=C0B�����Q�@�RA�{C;�                                    Bxf��   �          @�������>�@/\)A���C2�����ÿ��R@(�A�C@�                                    Bxf���  T          @θR����>aG�@�A��HC1�����׿u@Q�A�{C=\                                    Bxf��L  
�          @�z���>8Q�@
=A�G�C28R����G�@��A��\C=�H                                    Bxf���  
�          @��H���\>�(�@��A��
C/�{���\�W
=@A�33C<8R                                    Bxf�Ә  �          @˅����>W
=@Q�A���C1����Ϳ}p�@
�HA�C=��                                    Bxf��>  �          @�(���33���
?�\)A�{C4Ǯ��33��  ?�=qAep�C=G�                                    Bxf���  �          @��
��{?(�@�A�=qC.!H��{���@(�A��C9G�                                    Bxf���  �          @˅���R>L��@��A�(�C2����R�k�@ ��A�p�C<��                                    Bxf�0  �          @˅��33?O\)@\)A�ffC,
��33���@ffA���C7�q                                    Bxf��  �          @˅��\)?�  @��A��C'����\)��@0  A�
=C5W
                                    Bxf�+|  �          @�(����?�  @�A�
=C'����녽�\)@(��A�\)C4�R                                    Bxf�:"  �          @�=q��Q�?��?�{A���C.����Q��ff?��A��RC8E                                    Bxf�H�  �          @�����>.{?޸RA�  C2aH��녿5?���Aj=qC:�R                                    Bxf�Wn  �          @��H����>�z�?�p�A�C133���׿:�H?�\)A���C:�                                    Bxf�f  �          @�33����?�?���A�G�C.�q���׿�\?��HA��C8޸                                    Bxf�t�  �          @�  ��  ?\)@
=A��C.k���  �0��@z�A��HC:�=                                    Bxf��`  
�          @�  ���
>�Q�@�A���C0}q���
�8Q�?�p�A�{C;                                      Bxf��  �          @�G���  ?5@(�A���C,���  �
=@{A�33C9�f                                    Bxf���  �          @�Q����\?z�H@(��A�C)�q���\��@1�A�33C8��                                    Bxf��R  �          @�  ��Q�?&ff@z�A��RC-����Q�
=@A�  C9�)                                    Bxf���  �          @�  ��{?(�@{A���C-ٚ��{�333@��A�
=C;                                    Bxf�̞  �          @�ff����aG�@�A�{C<�������33?��Ah(�CF��                                    Bxf��D  �          @�
=��ff���@33A��C8��ff�Ǯ?�G�A��RCCL�                                    Bxf���  �          @�
=��{�#�
@p�A�\)C4p���{���\@
=A��C@�\                                    Bxf���  �          @ƸR���R>.{@�A�=qC2G����R���
@��A��
C>B�                                    Bxf�6  �          @�{��\)?p��@,(�AΣ�C*:���\)��@333A��C9z�                                    Bxf��  �          @�
=���?�\)@:�HA�{C%�\��녾�=q@Mp�A���C6�                                    Bxf�$�  �          @���>W
=@;�A�ffC1Ǯ�����
@(��Aʣ�CAB�                                    Bxf�3(  �          @���=q?0��@C�
A홚C,�\��=q�n{@@  A�Q�C=��                                    Bxf�A�  
�          @�����=q>���@C�
A�z�C0� ��=q��  @3�
Aٙ�CA(�                                    Bxf�Pt  
Z          @�ff����?��@>{A�C.(����Ϳ�G�@6ffA�p�C>��                                    Bxf�_  
Z          @�
=��p�>�33@AG�A���C0W
��p����H@2�\A�p�C@�
                                    Bxf�m�  T          @ƸR��ff>��@:�HA���C/)��ff���@0��AӮC>�R                                    Bxf�|f  
�          @�ff��ff?8Q�@4z�A�p�C,���ff�J=q@333A�C<E                                    Bxf��  �          @�p����\?W
=@�RA��C+h����\��\@$z�A��C9@                                     Bxf���  T          @�z���  >��@'
=A�z�C1Q���  ��=q@Q�A�Q�C?�                                    Bxf��X  
�          @�Q�����?�
=@I��A��\C �
���ý�@dz�B�
C5ff                                    Bxf���  U          @�=q����?�p�@7�A�
=C&T{������p�@FffA��C85�                                    Bxf�Ť  
�          @�p����@ff@^�RBp�C� ���>�z�@�{B4�RC0                                      Bxf��J  "          @�\)��ff@
=q@W�B	{C&f��ff>8Q�@�  B'��C1��                                    Bxf���  �          @�����G�?�{@O\)B�RC!\)��G��k�@g
=B{C6��                                    Bxf��  �          @������?��@Q�B��Cs3���ü#�
@q�BQ�C40�                                    Bxf� <  T          @���(�?.{@8��A�p�C,ff��(��\(�@5A�\)C=z�                                    Bxf��  �          @�����>#�
@Dz�A�Q�C2.������33@/\)Aܣ�CC�                                     Bxf��  
�          @�����H>��R@{A�Q�C0�����H�xQ�@33A�\)C>E                                    Bxf�,.  �          @�p���(��k�@�HA�p�C6h���(���z�?�p�A���CB��                                    Bxf�:�  "          @�{���
=L��@\)Aƣ�C3�����
���H@
�HA�Q�C@��                                    Bxf�Iz  �          @���33>��R@\)A�G�C0�3��33�z�H@�
A��C>ff                                    Bxf�X   �          @�=q���?aG�@+�A��C*������z�@0��A���C:0�                                    Bxf�f�  �          @�ff��G�?��
@B�\A�33C(����G��!G�@H��A�Q�C:�{                                    Bxf�ul  
Z          @Å���H?   @:=qA�  C.�3���H���
@1G�A�C>�3                                    Bxf��  
�          @�=q���?��H@3�
A�=qC'������Q�@A�A��C7��                                    Bxf���  �          @�G���{?�G�@9��A�C(���{�z�@AG�A�  C:\)                                    Bxf��^  	�          @������>�
=@8Q�A㙚C/xR������=q@,��A�Q�C?�\                                    Bxf��  T          @�Q�����?\)@N{B�C-�����ÿ��@Dz�A��
C@��                                    Bxf���  "          @����(�>���@@��A��C/�=��(���@333A�ffC@��                                    Bxf��P  T          @�  ��ff�9��?�(�A�p�CR^���ff�W�<�>�=qCVB�                                    Bxf���  T          @�����Q��E�?�z�AW�CS�{��Q��W������G�CU�                                    Bxf��  T          @�G����\�?\)?�p�A<z�CR�����\�Mp���G����RCT@                                     Bxf��B  T          @�����(��Vff?z�HACVs3��(��XQ�W
=� ��CV�q                                    Bxf��  "          @�����G��O\)@�
A�p�C<\)��G��޸R?�\)ATQ�CEk�                                    Bxf��  
�          @������\��=q@{A�p�C6Ǯ���\����?��
A�  CA�)                                    Bxf�%4  
�          @��H���R��@Q�A�Q�C@����R���?�(�A_�CI��                                    Bxf�3�  �          @�����������@'
=A��
C?n�����\)?��HA���CJ�H                                    Bxf�B�  
�          @�G���  ���?�Q�A�{CC���  ��
?fffA	G�CJ�=                                    Bxf�Q&  
Z          @Å����(�?���A�G�C@�����p�?}p�A�CG8R                                    Bxf�_�  �          @�ff������@\)Aȣ�C95�������?�z�A��CE�                                     Bxf�nr  
�          @�p������  ?�ffA�
CJ+������{�aG���CL�                                    Bxf�}  T          @��������?�z�Ay�CG5�����!G�>�ff@�ffCL:�                                    Bxf���  �          @�{��33��ff?��A��RCE����33� ��?:�H@���CL!H                                    Bxf��d  "          @�p����ÿ�33?�A�\)CF�R�����'
=?333@љ�CMB�                                    Bxf��
  
�          @�{��=q��(�?�G�A�ffCGn��=q�%�?�@��CL�)                                    Bxf���  �          @��R���H��(�?�Q�A_33CB�q���H���R?   @�=qCG�H                                    Bxf��V  
�          @�(���Q�Ǯ?�Q�A;33CCǮ��Q��Q�>u@
=CGk�                                    Bxf���  �          @����(��0��?�=qA4��C;B���(����?
=@��
C?�                                    Bxf��  �          @�(���Q��G�?�G�A$��C8����Q�aG�?&ff@�(�C=�                                    Bxf��H  "          @�=q����?�  ?�@�C$ٚ����?�G�?��RAH  C)�                                    Bxf� �  
�          @�����
=@<��>�G�@��CO\��
=@�?�{A��C��                                    Bxf��  T          @����@b�\?�@���C�����@1G�@G�A�
=CJ=                                    Bxf�:  T          @��~�R@{�?\(�AQ�CQ��~�R@=p�@.�RA�Q�CO\                                    Bxf�,�  �          @�  ���@���>���@B�\C8R���@U�@�\A���C��                                    Bxf�;�  T          @���Vff@�z�>���@UB����Vff@u@'�A�  C
                                    Bxf�J,  
�          @���J�H@��H>Ǯ@uB�W
�J�H@\)@1G�A��C p�                                    Bxf�X�  "          @��H�:=q@�G�>k�@{B��:=q@��@.�RA��
B��                                    Bxf�gx  T          @���  @�zᾏ\)�1�B�\)�  @���@{A�B�L�                                    Bxf�v  T          @�ff��@�������{�B����@��H@ffA��B�B�                                    Bxf���  
�          @�\)�(�@���=�Q�?Y��B�L��(�@�\)@7
=A��B�                                    Bxf��j  
�          @��!G�@�ff?!G�@�B�q�!G�@�@J=qB�B�=q                                    Bxf��  T          @�z�У�@��?O\)A�B�
=�У�@��\@]p�BG�B�.                                    Bxf���  "          @�(��p�@����Q�fffB�G��p�@��\@#�
A�  B�=                                    Bxf��\  
�          @�� ��@�Q�>\@p  B�{� ��@�=q@G
=A�G�B�                                    Bxf��  �          @����(�@��?(�@�\)B�  �(�@��
@P  B��B�33                                    Bxf�ܨ  T          @��Ϳ�p�@��R?uA�B��f��p�@�\)@e�B�B�(�                                    Bxf��N  �          @�
=�p�@�  ?(�@��B߸R�p�@�ff@R�\BG�B�Ǯ                                    Bxf���  "          @�G���z�@��?c�
A33B�\��z�@�{@g�BffB�W
                                    Bxf��  �          @������@�=q>��@}p�B�Ǯ����@�=q@R�\B��B�\                                    Bxf�@  T          @�33�}p�@��
>�  @=qB�\�}p�@�ff@J�HB ��Bʊ=                                    Bxf�%�  �          @�\)�\(�@�z�=�Q�?aG�BĸR�\(�@�G�@B�\A��B�z�                                    Bxf�4�  �          @�=q��{@��H?h��A�Bə���{@��H@n�RB33B�L�                                    Bxf�C2  "          @��׿}p�@�G�?c�
A
ffB�p��}p�@���@k�B\)B̔{                                    Bxf�Q�  �          @�p���G�@�  ?�33A�  BȸR��G�@s33@��BA(�BѨ�                                    Bxf�`~  �          @�p��c�
@���@�A�B�(��c�
@U�@�Q�BX33B��                                    Bxf�o$  "          @�
=�aG�@���@'�A���B�  �aG�@O\)@��B^�B�z�                                    Bxf�}�  �          @��Ϳ�p�@��
@�A�B�LͿ�p�@N�R@��
BQ�RB�G�                                    Bxf��p  �          @��R��@�Q�?aG�A
�RB噚��@�33@Z=qBG�B�G�                                    Bxf��  
�          @��N{@���@
=qA��B�z��N{@-p�@��\B0ffC��                                    Bxf���  �          @����@��H@{A���B螸���@Dz�@�33B@�
B��                                    Bxf��b  
�          @��R�%@���@A�ffB���%@<��@�p�BA��C@                                     Bxf��  
�          @�=q�   @�ff?+�@�(�B�=q�   @�(�@Tz�B�B�u�                                    Bxf�ծ  �          @�  �8��@�=�G�?�ffB�aH�8��@�p�@,��A�Q�B�aH                                    Bxf��T  T          @��H�0  @���>��@z=qB���0  @��@@��A�{B�ff                                    Bxf���  "          @Å�J=q@qG�@Z=qB
�C�q�J=q?�ff@�33BX  C��                                    Bxf��  �          @��H�_\)@�{?�z�AX(�B�G��_\)@Vff@dz�BC(�                                    Bxf�F  
�          @�
=�   @�Q�@  �陚B��f�   @�z�@�A�(�Bޅ                                    Bxf��  
�          @�녿��@�=q�����G�B����@�=q?k�A  B��                                    Bxf�-�  �          @�z���@qG��G
=��B��q��@�=q�L���
ffB➸                                    Bxf�<8  	�          @�G����?��
@ ��A�Q�C p����=u@<(�B	G�C3+�                                    Bxf�J�  T          @�Q���{?���@qG�B(�C����{���
@�B3G�C8Y�                                    Bxf�Y�  T          @��
���H@O\)@1�A�\)CL����H?�33@���B%C"
                                    Bxf�h*  �          @��
���@1G�@EA��CJ=���?Y��@��B&��C)J=                                    Bxf�v�  "          @����@[�@/\)Aң�C����?˅@�33B%��C�                                    Bxf��v  �          @�(���z�@q�@��A���C	����z�@�\@���B${C�\                                    Bxf��  �          @�����@�G�?�33A�G�C�)����@"�\@j�HB�RCB�                                    Bxf���  �          @�����@}p�@ ��A�ffC	(�����@�H@n�RB�
C}q                                    Bxf��h  T          @���  @~{@�A�C	  ��  @��@r�\Bp�C�3                                    Bxf��  T          @�p����H@�  @�\A���C�����H@�
@\)B!Q�C�                                     Bxf�δ  T          @�
=�1G�@���>�z�@.{B���1G�@���@B�\A�{B�{                                    Bxf��Z  �          @���XQ�@�G�>�  @G�B�\�XQ�@��R@7�A�33B�G�                                    