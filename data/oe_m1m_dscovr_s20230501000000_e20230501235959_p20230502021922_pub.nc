CDF  �   
      time          *   Conventions       ACDD-1.3, Spase v2.2.3     title         /DSCOVR Magnetometer Level 2 One Minute Averages    id        Doe_m1m_dscovr_s20230501000000_e20230501235959_p20230502021922_pub.nc   naming_authority      gov.noaa.swpc      program       DSCOVR     summary       }Interplanetary magnetic field observations collected from magnetometer on DSCOVR satellite - 1-minute average of Level 1 data      keywords      _NumericalData.ObservedRegion.Heliosphere.NearEarth, NumericalData.MeasurementType.MagneticField    keywords_vocabulary       Spase v2.2.2   
references        �; DSCOVR TIME SERIES DATA AVERAGES ALGORITHM THEORETICAL BASIS DOCUMENT, v2.4; GSE TO GSM COORDINATE TRANSFORMATION ALGORITHM THEORETICAL BASIS DOCUMENT v2.1      metadata_link         �http://www.ngdc.noaa.gov/docucomp/page?xml=NOAA/NESDIS/NGDC/STP/Space_Weather/iso/xml/satellite-systems_dscovr.xml&view=getDataView&header=none    license       Spase.Access Rights.Open   institution       NOAA   source        DSCOVR Magnetometer Level 1    platform      'Deep Space Climate Observatory (DSCOVR)    
instrument        +boom-mounted triaxial fluxgate magnetometer    history       ,DSCOVR real-time telemetry processing system   	algorithm         FDSCOVR MAGNETOMETER LEVEL 1B DATA ALGORITHM THEORETICAL BASIS DOCUMENT     algorithmVersion      B      algorithmDate         
2015-10-15     processing_level      Level 2    processing_level_description      11-minute average using Hodges-Lehmann M-estimator      date_created      2023-05-02T02:19:22.841Z   date_calibration_data_updated         2023-04-06T00:00:00.000Z   time_coverage_duration        P01D   time_coverage_start       2023-05-01T00:00:00.000Z   time_coverage_end         2023-05-01T23:59:59.000Z   time_coverage_resolution      PT1M   creator_name      Doug Biesecker     creator_type      person     creator_institution       DOC/NOAA/NWS/NCEP/SWPC     creator_email         doug.biesecker@noaa.gov    creator_url       http://www.swpc.noaa.gov/      publisher_name         National Geophysical Data Center   publisher_type        institution    publisher_institution         DOC/NOAA/NESDIS/NGDC   publisher_email       william.rowland@noaa.gov   publisher_url          http://www.ngdc.noaa.gov/dscovr/   records_maximum         �   records_present         �   records_data        �   records_fill             records_missing                    time                description       "date and time for each observation     
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
short_name        overall_quality    C_format      %d     units         n/a    lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale        7lBx}I��  �          @�  ?�ff�>�R�Vff�3�C�� ?�ff��{��33�3C�@                                     Bx}I�&  
�          @�  ?�ff�1G��Y���7\)C���?�ff�fff����C��
                                    Bx}I��  �          @��H?��R�9���S�
�-{C�8R?��R��������xC���                                    Bx}I�r  �          @�  �u�w��I���=qC�0��u�   ��=qz�C��H                                    Bx}I�  
�          @�G��\��Q��A���C�S3�\���������C��                                    Bx}J�  �          @�G�?fff��z��(���ffC�y�?fff�U��i���:��C���                                    Bx}Jd  �          @�33=��
��������G�C�w
=��
�`  �k��9��C���                                    Bx}J%
  �          @��Ϳ�\���\��Q����C�׿�\�aG��mp��8�RC��                                    Bx}J3�  �          @��>.{��ff���H�[\)C��q>.{�vff�U�#��C�Ff                                    Bx}JBV  �          @�z�>�����Q쿁G��>�\C�  >����qG��Dz���C��                                    Bx}JP�  T          @��>�Q����H�8Q��33C�q>�Q��~{�6ff�33C��3                                    Bx}J_�  �          @��\>�p����׿�R���C�q>�p����6ff���C���                                    Bx}JnH  �          @��>aG����\��p����RC�8R>aG���33�)������C�l�                                    Bx}J|�  T          @���=�����Q�����RC��\=�����  �,(��\)C���                                    Bx}J��  T          @�Q�#�
��
=�   ��{C�{�#�
��ff�-p��(�C��=                                    Bx}J�:  �          @�  �G�����@  ��C�.�G��p���1G���C��                                    Bx}J��  �          @�z�=�Q������(�����HC���=�Q��~{�1G���C��                                     Bx}J��  T          @�z�?@  ��  ��\�\C�}q?@  ��  �'��{C�Q�                                    Bx}J�,  �          @�{?�ff����?   @��C�5�?�ff���
����(�C��H                                    Bx}J��  �          @�
=?�������>��@��C�\)?�����G����
���\C��                                    Bx}J�x  T          @�
=?�33����>�@���C���?�33��Q���
��\)C��                                    Bx}J�  �          @��\?�ff���?
=q@أ�C�H�?�ff��(���  ����C��R                                    Bx}K �  �          @���?�{��G�?5A�
C�ٚ?�{�z�H�����}p�C�&f                                    Bx}Kj  �          @�=q?����z�H?n{AF=qC��=?����z=q�s33�K33C��                                    Bx}K  �          @��?�33���?�G�Aw�C���?�33��33�L����
C�c�                                    Bx}K,�  �          @�33?�����?��A�Q�C�5�?���
=�#�
���C��                                    Bx}K;\  �          @���?�{�x��?��A��C���?�{��=q�
=���C�O\                                    Bx}KJ  �          @�Q�?�\)��G�?\A��RC�ff?�\)��G��������C�                                    Bx}KX�  �          @���@XQ��3�
?���A���C�"�@XQ��U�>�33@�
=C���                                    Bx}KgN  
�          @�Q�@`���%�?�
=A�33C��R@`���J�H?\)@�ffC���                                    Bx}Ku�  �          @�@I���<(�?�=qA��C�|)@I���\��>���@l(�C�4{                                    Bx}K��  �          @��@L���:=q?�p�A�ffC��@L���W�>k�@6ffC��                                     Bx}K�@  �          @��@P  �1G�?�(�A�33C�˅@P  �P  >�\)@b�\C���                                    Bx}K��  �          @��H@HQ��-p�?�Q�AȸRC���@HQ��R�\?�\@��C���                                    Bx}K��  �          @��@o\)��p�@ffA�\)C��@o\)�.{?}p�A@z�C��                                    Bx}K�2  �          @�33@fff� ��@ ��Aϙ�C�o\@fff�,(�?c�
A2�RC��R                                    Bx}K��  "          @�(�@e�33@�AՅC�R@e�1G�?n{A9�C�(�                                    Bx}K�~  �          @�=q@i���
=q@33A�p�C��{@i���=p�?�=qAO\)C�|)                                    Bx}K�$  �          @���@e���@z�A�C�L�@e��?\)?��ARffC��                                    Bx}K��  S          @���@}p����@
=A�
=C�` @}p��z�?�Q�Ah��C��                                    Bx}Lp  �          @�  @��H���\@33A�z�C�^�@��H�33?�G�AvffC��=                                    Bx}L  �          @�\)@u��˅@\)A�C���@u���?��A~�\C���                                    Bx}L%�  �          @�
=@j�H��G�@A�
=C�.@j�H�'�?�ffA���C�9�                                    Bx}L4b  �          @���@X����@	��A�G�C�=q@X���4z�?}p�AK
=C�q                                    Bx}LC  �          @�p�@%�8��@�A��C���@%�_\)?�@��C�P�                                    Bx}LQ�  �          @�G�@S33�ff@�\A�C�E@S33�@��?E�AG�C�˅                                    Bx}L`T  "          @�=q@Tz��	��@  A�33C��@Tz��;�?�ffAV{C�Ff                                    Bx}Ln�  �          @�{@R�\� ��@�A��C�@ @R�\�1G�?�ffA\  C���                                    Bx}L}�  �          @�z�@Vff�\)@G�AΣ�C��3@Vff�HQ�?333A
�RC�|)                                    Bx}L�F  T          @���@J=q�
=@  A�G�C��R@J=q�G�?uAB�HC���                                    Bx}L��  "          @��@P  ��H@A��
C��{@P  �E?J=qA (�C�:�                                    Bx}L��  �          @���@QG���@ffA�ffC�Ff@QG��AG�?W
=A+
=C���                                    Bx}L�8  �          @��@AG��Q�?���AׅC���@AG��@  ?0��A�C���                                    Bx}L��  �          @��@8Q��1�?�A��HC���@8Q��S33>Ǯ@��C���                                    Bx}LՄ  T          @�(�@@���@��?�A�{C�}q@@���`  >�\)@`��C�W
                                    Bx}L�*  T          @�=q@S33�0  ?�z�A�p�C��@S33�E=#�
?\)C�n                                    Bx}L��  
�          @�@E�8Q�?��RA��
C�y�@E�^{?�\@�
=C��
                                    Bx}Mv  �          @�
=@I���:=q?�(�AŮC���@I���^�R>�@�=qC�
=                                    Bx}M  �          @�@W
=�&ff?��HA�Q�C�%@W
=�L��?(�@�C�5�                                    Bx}M�  �          @�\)@\��� ��?�33A�C���@\���E?
=@�G�C��                                    Bx}M-h  m          @�=q@e��@�A�{C�]q@e�>�R?���AY�C�                                      Bx}M<  m          @���@j=q��@{A�p�C��@j=q�8Q�?�ffAK�
C��                                     Bx}MJ�  �          @��@_\)�>�R?�=qAS\)C��@_\)�I����33���RC���                                    Bx}MYZ  �          @���@hQ��+�?��
A��C��@hQ��E�>B�\@C��q                                    Bx}Mh   �          @��\@��\�
=q?�A�ffC�\@��\�#�
>��
@s�
C��                                    Bx}Mv�  �          @�33@����=q?ٙ�A�
=C���@���
�H?Tz�A�RC�l�                                    Bx}M�L  T          @��\@�녿�?�p�A���C�W
@���\)?5A(�C�AH                                    Bx}M��  T          @�33@��H��33?�p�A��
C���@��H�{?8Q�A��C�q�                                    Bx}M��  �          @��@��\���R?�  A�z�C��)@��\�#33?333A�C���                                    Bx}M�>  "          @��\@�����?�=qA���C��R@���Q�?��@�C��                                    Bx}M��  T          @��@~�R�˅@�Aҏ\C�
@~�R�
=?��HAj=qC���                                    Bx}MΊ  T          @�G�@{���@(�A�Q�C��R@{���?�Q�A�p�C��3                                    Bx}M�0  "          @���@`�׿�z�@@  B{C�޸@`���ff@(�A���C�R                                    Bx}M��  �          @�=q@s�
�G�@5�B33C�4{@s�
��Q�@��A�=qC���                                    Bx}M�|  �          @��@p�׿h��@4z�BffC�/\@p���33@��A�ffC���                                    Bx}N	"  �          @��H@}p����@0��B(�C�&f@}p���
=@�A��HC�                                    Bx}N�  T          @���@|�;Ǯ@,(�Bz�C�0�@|�Ϳ\@  A�{C�w
                                    Bx}N&n  T          @��@qG����@>{B�RC���@qG����
@#�
B �RC���                                    Bx}N5  �          @���@qG��\)@;�B(�C��f@qG���33@$z�B\)C�˅                                    Bx}NC�  �          @���@s33=u@8Q�B�\?u@s33��Q�@'�B\)C�G�                                    Bx}NR`  T          @��@qG�>.{@>�RB�@$z�@qG����@0��B�C���                                    Bx}Na  �          @���@�G���@ ��A�=qC�˅@�G���{@  A�G�C�N                                    Bx}No�  �          @�\)@�Q콣�
@�RA�{C�k�@�Q쿓33@��Aޣ�C�H                                    Bx}N~R  �          @�{@xQ�?
=@#33B��A	p�@xQ��@#33B�C��=                                    Bx}N��  �          @�ff@]p�?�\@&ffB�A�
=@]p�>��@G
=B&��@���                                    Bx}N��  �          @��@qG�?��\@�RB��Ar�\@qG����@+�Bz�C��)                                    Bx}N�D  �          @��@s�
?�  @�A�ffAk
=@s�
���
@\)B�C�aH                                    Bx}N��  �          @���@a�?��H@z�A�  A��
@a�>��
@.{B{@���                                    Bx}Nǐ  T          @���@n{��\)@$z�B
G�C�o\@n{��@�A�RC�C�                                    Bx}N�6  �          @��@p�׿B�\@p�B��C�E@p�׿�  ?�33A�z�C���                                    Bx}N��  �          @��\@>{�   @
=A�=qC��=@>{�QG�?�ffAVffC��                                    Bx}N�  �          @��
@=p��#�
@p�B ��C���@=p��W�?�\)A`Q�C��                                     Bx}O(  �          @��@X����@  A�p�C���@X���;�?���A\Q�C��{                                    Bx}O�  �          @��
@\(��
�H@��A�C��H@\(��9��?��ATz�C���                                    Bx}Ot  T          @�33@O\)�p�@��A��\C��{@O\)�AG�?��HAtQ�C�~�                                    Bx}O.  �          @��\@U���
=@(�B��C��)@U��2�\?���A�p�C��                                    Bx}O<�  �          @��\@R�\����@%B	�C�O\@R�\�1�?�ffA�{C��f                                    Bx}OKf  T          @��H@R�\��=q@%B
{C�p�@R�\�0��?ǮA���C��q                                    Bx}OZ  
�          @��@Q녿�@'
=B\)C�� @Q��(Q�?�33A�\)C��=                                    Bx}Oh�  
�          @��H@Q녿�p�@5B=qC��q@Q��#33?�Q�A�ffC�3                                    Bx}OwX  "          @��@N{��=q@3�
BffC��3@N{�(Q�?�\)A�(�C�h�                                    Bx}O��  T          @��@XQ쿺�H@,(�Bp�C�K�@XQ��{?�A��C��                                    Bx}O��  T          @��\@^{����@*�HB��C��@^{��?�{A���C��                                    Bx}O�J  �          @���@^�R�}p�@/\)B�\C��@^�R��
@�
A�Q�C��
                                    Bx}O��  �          @�33@k��fff@%B	�\C�%@k���z�?�(�A��C�E                                    Bx}O��  T          @�Q�@W��
�H@33AظRC���@W��5?p��A@z�C���                                    Bx}O�<  
�          @�Q�@P  �{@(�A���C��)@P  �<(�?�ffAW�C���                                    Bx}O��  "          @���@G
=�33@A�\)C���@G
=�Dz�?�33Ak�
C���                                    Bx}O�  �          @���@I���33@�A�\C��3@I���C33?���A`��C��q                                    Bx}O�.  
�          @���@G��(�@�RA���C��
@G��J=q?}p�AIp�C�W
                                    Bx}P	�  T          @���@Tz��ff?���Ạ�C�e@Tz��<��?G�A
=C�8R                                    Bx}Pz  T          @��H@O\)�Q�@p�A�ffC�޸@O\)�E?�  AJffC�4{                                    Bx}P'   �          @�33@E��p�@�A��RC���@E��N{?�=qAZffC��                                    Bx}P5�  T          @�(�@Fff�z�@!�B�C��\@Fff�J�H?���A��HC�(�                                    Bx}PDl  T          @�p�@Dz���
@(��B
G�C��f@Dz��L��?�
=A�\)C��H                                    Bx}PS  �          @��@I����@#33B�C��@I���H��?�{A�z�C���                                    Bx}Pa�  T          @��@O\)�33@�A��C�Y�@O\)�Fff?�  Ax��C�"�                                    Bx}Pp^  "          @�z�@QG��z�@�A���C�O\@QG��E�?��Ad��C�Y�                                    Bx}P  �          @��@XQ��
=@A��C���@XQ��8��?��RAyp�C���                                    Bx}P��  �          @���@\(��
=?��HA�=qC�Ф@\(��6ff?�@�G�C�,�                                    Bx}P�P  �          @�  @`  ���?�Q�A�{C�Ф@`  �1�>��R@�Q�C��=                                    Bx}P��  T          @���@`���(�?��A�Q�C��)@`���1�?O\)A#33C�˅                                    Bx}P��  �          @�=q@dz��z�?�Q�A�
=C��\@dz��,(�?fffA6ffC��H                                    Bx}P�B  "          @��H@aG���Q�@p�A��HC��{@aG��+�?��HAt��C�XR                                    Bx}P��  �          @��\@\�Ϳ�@z�A�G�C�p�@\���-p�?��A�\)C��                                    Bx}P�  
�          @���@]p���
=@�RA�\C�l�@]p��+�?�p�Az�RC�"�                                    Bx}P�4  �          @��H@fff��(�@33Aԣ�C���@fff�(��?��AU�C��                                    Bx}Q�  T          @��H@g��z�?�33A�{C�R@g��*�H?aG�A0Q�C��                                    Bx}Q�  
�          @�Q�@`  �
=q?���A��C�'�@`  �.�R?J=qA"=qC��                                    Bx}Q &  �          @���@^{��?��HAͅC�J=@^{�/\)?h��A:{C��q                                    Bx}Q.�  �          @���@aG��{?��HA��HC��q@aG��.�R?&ffA(�C��                                    Bx}Q=r  "          @�  @\��� ��?��RA�Q�C��@\���)��?}p�AMC�>�                                    Bx}QL  
�          @��
@P�׿�Q�@:�HB33C��@P��� ��@z�A�p�C�,�                                    Bx}QZ�  �          @�33@U����H@3�
B�RC�  @U���R?��HA�=qC���                                    Bx}Qid  �          @��H@a녿�@��A���C�W
@a�� ��?�G�A��C�N                                    Bx}Qx
  �          @��H@hQ����@Q�A�\)C��{@hQ��!�?���Ar�HC���                                    Bx}Q��  �          @�
=@J=q��@*�HB��C��@J=q�'
=?�  A��C�.                                    Bx}Q�V  �          @�@O\)��ff@�HBp�C�q�@O\)�(Q�?�(�A�p�C�s3                                    Bx}Q��  
�          @�z�@@  ��p�@+�B�HC���@@  �*�H?޸RA�\)C�!H                                    Bx}Q��  T          @�p�@AG���(�@#33B�
C�w
@AG��5�?��
A��C�l�                                    Bx}Q�H  
�          @���@7
=��\@=qB{C���@7
=�Dz�?��\A�ffC�|)                                    Bx}Q��  �          @���@I����@��A�C���@I���7
=?���Ab{C��                                     Bx}Qޔ  T          @�Q�@o\)�\@   Aҏ\C��@o\)���?��HAyG�C���                                    Bx}Q�:  �          @���@j=q����?�z�Aȏ\C�˅@j=q�(�?�G�AO�C�.                                    Bx}Q��  
�          @�
=@[���p�@�A��C��q@[�� ��?��A���C��\                                    Bx}R
�  
�          @�{@R�\��z�@�A�C��=@R�\�*�H?���A�=qC�u�                                    Bx}R,  T          @�Q�@e�����@p�A�p�C���@e��
=?���A��HC�W
                                    Bx}R'�  
�          @�
=@Y����=q@��A�
=C��H@Y���%�?��A���C�c�                                    Bx}R6x  T          @�{@R�\��=q@A�G�C�s3@R�\�'�?�z�A�Q�C��q                                    Bx}RE  �          @�z�@L(���\)@Q�B��C�ٚ@L(��*=q?�
=A��C��                                    Bx}RS�  T          @�@Tz����@33A���C��
@Tz��%?���A�
=C���                                    Bx}Rbj  
�          @���@_\)��\)@	��A�  C��f@_\)�%�?�(�A{�C���                                    Bx}Rq  
�          @��R@]p�����@Q�A�=qC�*=@]p�� ��?�(�A
=C��                                    Bx}R�  �          @�ff@b�\�У�@
=A�C��@b�\�?��A���C�J=                                    Bx}R�\  �          @��@a녿�Q�@�A�
=C�1�@a���H?�=qA��C��f                                    Bx}R�  �          @�Q�@i�����H@ ��A�z�C�n@i����?�Ap(�C�z�                                    Bx}R��  �          @�Q�@w���ff?��HA���C��@w��ff?uAD  C��                                     Bx}R�N  �          @��@h�ÿ�
=?�(�A�p�C�� @h���z�?��Am�C���                                    Bx}R��  �          @�Q�@l(�����@�A�ffC�N@l(��G�?�p�A}�C�33                                    Bx}Rך  �          @�G�@l�Ϳ���@ffA�=qC�}q@l���G�?�ffA��C�7
                                    Bx}R�@  "          @�p�@g��У�?�A�
=C��)@g����?�\)Al  C�H                                    Bx}R��  
�          @�  @x�ÿ��?޸RA���C�0�@x�ÿ��H?�ffAX��C��H                                    Bx}S�  
�          @��@z=q��ff?�=qA�(�C�.@z=q�33?W
=A+�C�0�                                    Bx}S2  T          @�Q�@��׿�ff?��A�Q�C�n@��׿��H?�R@��C�                                    Bx}S �  "          @��@~�R��Q�?���A�p�C��@~�R���?E�AffC�G�                                    Bx}S/~  �          @���@��׿�Q�?�ffA�  C�+�@��׿�
=?^�RA/�C�,�                                    Bx}S>$  T          @�Q�@~�R��(�?��RA��C���@~�R��Q�?L��A#\)C��                                    Bx}SL�  �          @���@��ÿ���?�\)A�{C�/\@���� ��?!G�A (�C��                                    Bx}S[p  �          @���@~{����?�(�A�C�H@~{�33?:�HA  C�W
                                    Bx}Sj  
�          @��@w
=�У�?ǮA�G�C���@w
=�
=?L��A$��C��\                                    Bx}Sx�  "          @��@y���Ǯ?�ffA�=qC�)@y����\?Q�A)G�C�0�                                    Bx}S�b  �          @�Q�@~{����?��HA���C�<)@~{� ��?:�HA��C���                                    Bx}S�  �          @���@u���G�?��A�Q�C���@u����?Tz�A)G�C��H                                    Bx}S��  T          @��@w
=��G�?��A�C�Q�@w
=��?�33Aj�RC��R                                    Bx}S�T  T          @��\@����?��RA}�C�!H@���(�?!G�@�p�C�ٚ                                    Bx}S��  �          @�Q�@�33���?�  A�p�C�*=@�33��Q�?&ffA��C�Ф                                    Bx}SР  �          @��@fff� ��?޸RA���C�n@fff�!�?Tz�A*�RC�}q                                    Bx}S�F  �          @���@k���\?��A��HC�0�@k��
=?��AZ�HC���                                    Bx}S��  "          @���@x�ÿ�\?��HA���C��H@x�����?+�A�C�C�                                    Bx}S��  T          @���@��ÿ�z�?��A[�C���@��ÿ�Q�>�33@�p�C�&f                                    Bx}T8  �          @�{@�(���
=?:�HA��C�k�@�(���{>�?�33C�]q                                    Bx}T�  T          @��@����{?=p�A�
C���@�����>.{@
=qC���                                    Bx}T(�  �          @�(�@������>�@��C�R@����z�u�B�\C��\                                    Bx}T7*  �          @�(�@��H��z�?(�A (�C��H@��H���
<��
>��
C���                                    Bx}TE�  �          @�33@�=q��=q?@  A�
C��R@�=q��G�>B�\@!G�C�˅                                    Bx}TTv  �          @�33@~�R��=q?8Q�A�
C�,�@~�R��p�=u?G�C�:�                                    Bx}Tc  �          @��@��H���R>L��@)��C��{@��H��(���z��w
=C���                                    Bx}Tq�  �          @�33@�����\)?�RA�
C���@�����  =L��?333C��
                                    Bx}T�h  �          @��@~�R��33?��@�{C��@~�R��  ��\)�fffC�!H                                    Bx}T�  �          @��@�33��  ?�@��HC��@�33�˅���
����C�h�                                    Bx}T��  �          @�@����\)=�\)?s33C�>�@�����
�
=q��\C��\                                    Bx}T�Z  �          @�ff@��ÿ��
?�@�\C�{@��ÿ�{�����33C��
                                    Bx}T�   T          @�
=@�Q���?   @���C�s3@�Q��
=��  �K�C�*=                                    Bx}Tɦ  �          @��@y����>�?�G�C���@y����=q�����p�C�n                                    Bx}T�L  �          @�@��׿��;���\)C��)@��׿��O\)�)G�C��{                                    Bx}T��  T          @��@~{����33��33C��3@~{���ÿ��\�V{C�1�                                    Bx}T��  �          @�(�@z=q��{�#�
��C�G�@z=q��p�������C��)                                    Bx}U>  �          @��@fff�{�p���G�C�33@fff���H�ٙ���z�C�T{                                    Bx}U�  �          @�=q@c�
�녿�ff���\C�*=@c�
��33���H���C�H�                                    Bx}U!�  �          @���@e���H��p���{C��\@e��{��\)��{C��H                                    Bx}U00  �          @���@`  �ff��{���C�� @`  ��Q��33��p�C��=                                    Bx}U>�  �          @�
=@N�R�\)�z�H�Y�C�/\@N�R��Q�����C�w
                                    Bx}UM|  T          @r�\@$z��p������C��@$z�n{�ff�"C�)                                    Bx}U\"  T          @l��?�
=�333�L���q�C��{?�
=?
=q�N�R�u��A��                                    Bx}Uj�  "          @|��?�����33�i��#�C��f?���?��\�aG��B!                                    Bx}Uyn  �          @�z�@3�
��Q��+��%
=C���@3�
���;��8��C���                                    Bx}U�  �          @��
@,���#33��=q��(�C�XR@,�Ϳ������\)C�C�                                    Bx}U��  �          @��@p��,�Ϳ�
=��33C�*=@p���33�#33��C�1�                                    Bx}U�`  �          @�33@K��
=�B�\�-�C���@K���z��������C�|)                                    Bx}U�  T          @�z�@Vff����u�Tz�C�7
@Vff��ÿ����w\)C��{                                    Bx}U¬  T          @�@H���(�ÿ(���{C��{@H���{��=q����C�`                                     Bx}U�R  �          @�\)@i�������aG�C�  @i�����Ϳ�  �\Q�C��3                                    Bx}U��  �          @�  @r�\��׽��Ϳ�C��{@r�\���H�G��*ffC��q                                    Bx}U�  �          @�
=@fff�������C�� @fff�����  ����C���                                    Bx}U�D  �          @��H@l(���R<��
>���C�w
@l(����J=q�(  C�J=                                    Bx}V�  T          @���@u�333?�Q�A��C��q@u����?�ffA�
=C�O\                                    Bx}V�  T          @�Q�@n�R�\(�@�
A��\C��f@n�R�У�?�A�ffC�+�                                    Bx}V)6  �          @��H@O\)���@O\)B3�
C��{@O\)���@>{B!\)C��                                    Bx}V7�  �          @��
@G�=#�
@E�B2ff?@  @G�����@8Q�B$��C���                                    Bx}VF�  �          @�p�@k���@\)A���C�@k��k�@33A���C���                                    Bx}VU(  �          @�  @l���33?h��A<(�C��@l���{<��
>�  C�!H                                    Bx}Vc�  �          @��@]p��$z�?���AyC��)@]p��5�>L��@&ffC�c�                                    Bx}Vrt  �          @��@hQ��0  ?���A�=qC�l�@hQ��N{?5A�C�5�                                    Bx}V�  �          @�\)@P���'
=@%�A�
=C���@P���W
=?�(�A�\)C�{                                    Bx}V��  �          @��@-p��0  @S33B"�C�Ff@-p��p  @ffA�p�C���                                    Bx}V�f  �          @��\@0���C�
@<(�BG�C��@0���z=q?�z�A�G�C���                                    Bx}V�  T          @��\@<���;�@9��B��C��
@<���qG�?�A�C��                                    Bx}V��  �          @��\@8���E@2�\Bz�C���@8���xQ�?�G�A�z�C�U�                                    Bx}V�X  T          @���@1��QG�@)��A�p�C�,�@1��\)?���ArffC�g�                                    Bx}V��  �          @��H@.{�E@=p�B��C��f@.{�|(�?�
=A�z�C�K�                                    Bx}V�  �          @��@7��P��@)��A���C��@7��~�R?�=qAqC��H                                    Bx}V�J  
�          @��H@�
�P  @L(�B�RC���@�
��p�?�A��C��                                    Bx}W�  �          @�@33�\(�@FffBz�C�޸@33���?�Q�A���C�f                                    Bx}W�  �          @�@Q��^�R@K�B�RC���@Q����
?�  A��RC�H                                    Bx}W"<  �          @�p�?�Q��c33@W
=B"{C���?�Q���  ?�33A��\C�AH                                    Bx}W0�  �          @�p�?��j�H@J=qBQ�C�S3?���G�?�
=A�
=C��                                    Bx}W?�  �          @���@��q�@1G�Bz�C��@���Q�?��
Af=qC��
                                    Bx}WN.  �          @��?�z��u@0��B��C�8R?�z����?�  Ab=qC�\)                                    Bx}W\�  T          @��
?��H����@(��A�
=C���?��H��?���AB�\C��                                    Bx}Wkz  �          @�z�?Ǯ��p�@#33A��C�Ff?Ǯ����?p��A(z�C��                                    Bx}Wz   �          @�33?�{�}p�@&ffA�\)C���?�{���
?��A@z�C��
                                    Bx}W��  
�          @�ff@z��j�H@
=A�(�C�&f@z���Q�?p��A0(�C�J=                                    Bx}W�l  �          @��
@1G��_\)?�Q�A�ffC�:�@1G��|��?�R@陚C���                                    Bx}W�  �          @��@-p��dz�?�\)A��C��
@-p��\)?�@�{C��                                    Bx}W��  
(          @�(�@)���n�R?�z�A�G�C��@)����=q>�=q@L��C���                                    Bx}W�^  "          @���@
=q��=q?�Q�Ag�
C���@
=q��\)��\)�W
=C���                                    Bx}W�  �          @�(�@
�H�x��?��RAy��C���@
�H���\�#�
��\C�H                                    Bx}W�  �          @��@(Q��i��?�p�A�33C��f@(Q��|(�>\)?�z�C���                                    Bx}W�P  	�          @��R@z��xQ�?�G�Ay�C�p�@z����\�\)��C���                                    Bx}W��  "          @��R@5�U?�A��\C�+�@5�mp�>�(�@��C��{                                    Bx}X�  �          @���@7
=�_\)?�G�A�ffC��=@7
=�s33>u@6ffC�~�                                    Bx}XB  "          @���@=q����?s33A5��C�ff@=q��(�������C�&f                                    Bx}X)�  �          @���@!��s�
?��A�  C��3@!����ýu�+�C�R                                    Bx}X8�  "          @���@>�R�HQ�?ǮA�ffC��=@>�R�^{>��@�=qC�P�                                    Bx}XG4  
�          @�(�@p  ��?�p�A�
=C��@p  ���?��RAy��C���                                    Bx}XU�  T          @��\@~�R���@ffA�
=C�\@~�R���?�A��C�T{                                    Bx}Xd�  �          @���@~�R���@�AڸRC�
=@~�R��(�?�\A�Q�C�z�                                    Bx}Xs&  �          @��@�z��?���A��
C���@�z�.{?�=qA�ffC�W
                                    Bx}X��  T          @�(�@w����@
=qA�ffC�y�@w��ٙ�?�
=A�C�                                      Bx}X�r  �          @�(�@vff�0��@p�B z�C��@vff��(�@�A�z�C���                                    Bx}X�  �          @��H@�33��{?��A�C��)@�33�J=q?���Ar�HC�ٚ                                    Bx}X��  �          @���@�{�#�
?��A��C��R@�{���?���AqG�C���                                    Bx}X�d  T          @���@~�R�Q�@�
AָRC�4{@~�R����?�A��C���                                    Bx}X�
  �          @�{@|��=#�
@\)B ��?!G�@|�ͿG�@
=A��C�b�                                    Bx}Xٰ  �          @�ff@xQ쾙��@)��B	
=C��\@xQ쿔z�@��A��C���                                    Bx}X�V  �          @�@|(��!G�@�HA��
C�s3@|(���33@�A�G�C�AH                                    Bx}X��  "          @�\)@q녿fff@+�B
�C�T{@q녿�p�@\)A�Q�C��3                                    Bx}Y�  �          @�  @u���\@%B(�C���@u����@
=A��HC�W
                                    Bx}YH  T          @�@vff��=q@33A�C��H@vff� ��?޸RA���C�<)                                    Bx}Y"�  T          @���@y����G�@�A�C��@y����z�?�33A��C��q                                    Bx}Y1�  
�          @�33@|�Ϳ��@A�(�C�P�@|�Ϳ��H?�\)A���C�L�                                    Bx}Y@:  �          @�@�녾u@�A�(�C�H�@�녿h��?��A¸RC��{                                    Bx}YN�  �          @�\)@w
=?���@��A��\A��\@w
=>�z�@(��B	{@�                                      Bx}Y]�  �          @�
=@���?(��@
=A�Q�A��@��þ\)@��A�(�C��q                                    Bx}Yl,  T          @�  @��R�#�
@�
AθRC��{@��R���\?޸RA��C���                                    Bx}Yz�  T          @�=q@����
=?��A��RC���@����?�G�AC
=C�}q                                    Bx}Y�x  �          @���@vff�,(�?333A(�C��@vff�1G��L���(�C�%                                    Bx}Y�  �          @�(�@l���/\)>.{@
=C��R@l���)���8Q��z�C�/\                                    Bx}Y��  �          @��@qG��!�=���?���C�
=@qG����8Q���HC��
                                    Bx}Y�j  �          @�  @W��<�;�Q����C�b�@W��+������\)C��                                     Bx}Y�  �          @��H@XQ��A�>��
@\)C��@XQ��>{�+���C�W
                                    Bx}YҶ  T          @�
=@dz��7�?��
AK\)C���@dz��C33=u?B�\C���                                    Bx}Y�\  �          @���@n�R�!G�?�\)A^�HC��q@n�R�/\)>��@I��C���                                    Bx}Y�  �          @��@vff�
=?�G�A�(�C��=@vff��R?E�A�C���                                    Bx}Y��  T          @�(�@}p���ff?��
A��C�Ǯ@}p��(�?aG�A0��C��H                                    Bx}ZN  
�          @��H@���u?��HA��\C�o\@������?���A��RC�Ff                                    Bx}Z�  T          @��@}p��O\)@A�ffC�>�@}p���
=?�(�A�(�C��                                    Bx}Z*�  T          @�G�@tz�>���@�A�G�@�\)@tzᾮ{@�\A�ffC�w
                                    Bx}Z9@  T          @�G�@R�\@p�@
=A��
B=q@R�\?�
=@,(�Bz�A��
                                    Bx}ZG�  �          @��@L(�?�\)@�RB=qA��@L(�?�  @<(�B%G�A�
=                                    Bx}ZV�  T          @�  @Vff@{?�A�{B�@Vff?�G�@!G�B	��A�                                    Bx}Ze2  
�          @�G�@
=q@P  @ffA��HBa��@
=q@�@@��B+�BAz�                                    Bx}Zs�  �          @��H@(Q�?�=q@HQ�B1z�B�\@(Q�?B�\@c33BQz�A��                                    Bx}Z�~  �          @��H?��
?(�@\)B�  A��?��
�(��@~�RB�u�C�W
                                    Bx}Z�$  �          @��
?ٙ�@s33?�ffA��RB���?ٙ�@C33@8��BQ�Bs��                                    Bx}Z��  T          @�z�?��
@aG�@��A���B|��?��
@&ff@XQ�B;�B^��                                    Bx}Z�p  ;          @�33@z�@Q�@*�HBQ�BZ��@z�@�\@c33B=��B2ff                                    Bx}Z�  ;          @�
=@�H@H��@#33B33BQ�R@�H@��@Y��B8�B)33                                    Bx}Z˼  T          @�z�@#�
@��@<(�B�B.@#�
?�@c33BI��A�ff                                    Bx}Z�b  T          @��\@{@!G�@#33B�B6(�@{?�\)@L��B=  B��                                    Bx}Z�  
�          @�{@l(��#�
@)��Bz�C�Ǯ@l(��u@{BC��3                                    Bx}Z��  T          @��
@u�=p�@�A��\C��f@u����@A�G�C���                                    Bx}[T  ;          @�\)@p  �=p�@�A�33C�b�@p  ��@   AӅC���                                    Bx}[�  �          @�(�@{����\@�
A�Q�C�3@{���?�=qA�  C�}q                                    Bx}[#�  m          @��@y�����H?�G�A�{C���@y�����?�=qAT��C�(�                                    Bx}[2F  	          @��@q��
=?޸RA�33C�q�@q��!�?��\AI�C�
                                    Bx}[@�  T          @�33@aG��'
=?���A�  C��)@aG��9��?�@��HC�=q                                    Bx}[O�  m          @��
@y���У�?�ffA�ffC��f@y���ff?�(�Av�RC��)                                    Bx}[^8  m          @���@s�
���?�Q�A�z�C�g�@s�
�{?���AYC��                                    Bx}[l�  �          @�z�@g
=� ��?���A�ffC���@g
=�5�?!G�@�33C���                                    Bx}[{�  �          @��H@\���2�\?�z�Aj=qC���@\���@��>�=q@W
=C�q�                                    Bx}[�*  �          @��@<���L(�?(��A
�RC�j=@<���O\)���
��ffC�/\                                    Bx}[��  T          @�p�@2�\�X��>��@���C��
@2�\�W
=�!G���\C���                                    Bx}[�v  T          @�=q@+��W�>���@���C�=q@+��U��R���C�b�                                    Bx}[�  
�          @�
=@#33�U�>�  @VffC��3@#33�P  �B�\�%�C�f                                    Bx}[��  T          @�G�@7
=�5?!G�A{C��3@7
=�9���u�Y��C�G�                                    Bx}[�h  �          @��
@W
=�:�H?�(�AtQ�C��@W
=�I��>���@l(�C�q�                                    Bx}[�  �          @�G�?޸R�^�R���{C�C�?޸R�J=q��=q����C�k�                                    Bx}[�  T          @~�R?У��`�׿Y���F�\C�}q?У��Fff������Q�C��                                    Bx}[�Z  "          @|��?���l�Ϳ333�$z�C��=?���U���G���ffC�l�                                    Bx}\   "          @��?=p��w������p�C�p�?=p��W
=��R��\C�:�                                    Bx}\�  �          @���?��l(��Ǯ���C�j=?��Dz��$z���C�G�                                    Bx}\+L  �          @��>�(��s33��z���C�=q>�(��N{����G�C��\                                    Bx}\9�  �          @�p��aG��b�\�����ffC�4{�aG��5��5�3C�                                    Bx}\H�  �          @����ff���C33�?p�Cq�3��ff��Q��g��wQ�Cd\                                    Bx}\W>  �          @��Ϳ���{�J=q�E��Cv&f��������o\)G�Ch��                                    Bx}\e�  
�          @\)��
=�Fff���\)C�)��
=����N{�YC��{                                    Bx}\t�  T          @�G�����L(��(����C�L;���ff�P���W(�C�                                      Bx}\�0  �          @�  ?���s�
�xQ��[33C�S3?���W��G���33C�XR                                    Bx}\��  �          @�\)@XQ��O\)?�Q�A��\C�{@XQ��e?.{@��C���                                    Bx}\�|  "          @�
=@A��e?˅A�{C��@A��y��?   @���C��                                    Bx}\�"  �          @�{@B�\�hQ�?�p�Ai��C��{@B�\�u�>\)?�Q�C�33                                    Bx}\��  �          @��?Ǯ��ff�W
=�)�C�5�?Ǯ�q��   ��z�C�8R                                    Bx}\�n  "          @��>��H���ÿc�
�9C�K�>��H�vff�z���(�C���                                    Bx}\�  �          @��R>����G���{�u�C�޸>���c�
�p����C�AH                                    Bx}\�  �          @��?
=���������c�C���?
=�tz��G���  C�g�                                    Bx}\�`  �          @�=q=���G���  ��(�C��3=��l(��)����C��{                                    Bx}]  
�          @�=q�fff�k���R�  C�'��fff�5��Z�H�D�C|p�                                    Bx}]�  �          @�
=�.{��33��
����C��{�.{�Q��W��6�RC��=                                    Bx}]$R  �          @�=q�Ǯ�n{�-p���HC�  �Ǯ�3�
�h���P�\C�
=                                    Bx}]2�  �          @���=�����Q���R����C��=����l���Z=q�*p�C��=                                    Bx}]A�  T          @��=#�
��ff��R��33C�AH=#�
�i���X���+p�C�P�                                    Bx}]PD  �          @�Q�>�p���(������RC�J=>�p��{��C33���C���                                    Bx}]^�  �          @��
?   ���Ϳ����|  C��?   ��33�#�
��=qC�s3                                    Bx}]m�  �          @��>�����Ϳ��R��(�C��3>������.{�ffC�
                                    Bx}]|6  T          @�(�=L����ff��G��o
=C�U�=L����p�� �����C�`                                     Bx}]��  �          @��H>�\)�����z���p�C�>�\)��G��'����C�H                                    Bx}]��  T          @�\)?�����=q�\���\C���?�����G���\)���HC��)                                    Bx}]�(  �          @��?�Q���\)�\��(�C��)?�Q���{�����RC�p�                                    Bx}]��  �          @�ff?�(���{�c�
�&�\C��?�(������z��ƸRC��                                    Bx}]�t  T          @�Q�?�\)��G��s33�/
=C�+�?�\)����	����  C�5�                                    Bx}]�  �          @��R?�
=��=q�Y����C�R?�
=�����
���C��)                                    Bx}]��  �          @�=q?�z���������C�C�O\?�z������z���(�C�*=                                    Bx}]�f  T          @���?�����p������J�\C���?�����ff�z�����C��=                                    Bx}^   �          @���?�  ������nffC�y�?�  ���� ����{C�]q                                    Bx}^�  T          @���?���������\�;�C��?�����G������{C���                                    Bx}^X  �          @�=q?J=q��{�L���G�C���?J=q��G��ff���
C��
                                    Bx}^+�  
�          @�=q?�{��(��:�H�G�C�e?�{��  �G����C��                                    Bx}^:�  �          @�Q�?����(����\�=p�C�0�?����{�{��(�C�
                                    Bx}^IJ  T          @�\)?�(����\�p���.�HC�P�?�(�����Q����HC�@                                     Bx}^W�  �          @��R?��H�������G�
C�7
?��H��\)����\)C��{                                    Bx}^f�  �          @��?˅��z�W
=�z�C�u�?˅��  �33��ffC�B�                                    Bx}^u<  �          @�?����\)�Tz��ffC�Z�?����33���R��{C�H�                                    Bx}^��  �          @��R?޸R��녿8Q��ffC�xR?޸R���R��33��G�C�C�                                    Bx}^��  �          @��@%���c�
�#�
C��@%�s33���H����C�#�                                    Bx}^�.  �          @���?�33��Q�\(��!��C��?�33���
� ����
=C��                                    Bx}^��  �          @��R>�Q����H�G��C�=q>�Q���
=���H��=qC�o\                                    Bx}^�z  T          @��>����33�(���=qC��{>����Q�����\C��                                    Bx}^�   �          @��R������=q��{�[
=C�c׽������
�G����C�S3                                    Bx}^��  �          @��>W
=��Q�   ���C�U�>W
=��\)��z���G�C�l�                                    Bx}^�l  �          @��\=L�����ÿ�R���C�S3=L����
=���
��
=C�Y�                                    Bx}^�  
�          @�z�?�p������=q��{C��?�p��p  �Q���
=C�3                                    Bx}_�  �          @��R?k����ÿ�p����C�q�?k��`���{�	=qC�]q                                    Bx}_^  �          @��\?��\����(��x��C�k�?��\�n�R�����z�C�aH                                    Bx}_%  T          @�Q�?���\)��=q����C�C�?���X���2�\��C�Ǯ                                    Bx}_3�  �          @�z�?z�H�s33�7��C�7
?z�H�;��p  �I�C�5�                                    Bx}_BP  �          @���?�  ������ә�C�?�  �Z=q�J=q�!  C�ٚ                                    Bx}_P�  T          @�(�?������Ϳ�z���Q�C��?����e�*�H���C���                                    Bx}__�  �          @��@�\���H��Q����\C��)@�\�e�(���(�C�B�                                    Bx}_nB  �          @�@����H��G��n{C�W
@��hQ��������C��q                                    Bx}_|�  �          @�p�@%��|�Ϳ�\)���C���@%��^�R����\C�G�                                    Bx}_��  "          @���@.{�~{��Q���G�C�:�@.{�^�R�����  C�
=                                    Bx}_�4  �          @�33@.�R�|(��޸R��  C�c�@.�R�W��+���C���                                    Bx}_��  �          @�33@9���j=q��p����C�8R@9���B�\�5�{C���                                    Bx}_��  �          @��@#33���ÿ���=G�C�c�@#33�xQ�����\)C��f                                    Bx}_�&  �          @�\)@AG���p��Q��p�C���@AG��u�����p�C�%                                    Bx}_��  
�          @�p�?�(���>�{@qG�C�N?�(�����fff�!p�C�k�                                    Bx}_�r  �          @��
?\���
>�\)@FffC���?\��G��s33�*�RC��\                                    Bx}_�  
�          @���?\����=�\)?Tz�C�Ф?\���Ϳ�\)�N{C��                                    Bx}` �  �          @��?�  ���<�>ǮC�(�?�  ��33����P��C�p�                                    Bx}`d  
�          @�  @Q����R�\)��  C���@Q�����33����C��                                     Bx}`
  
�          @���@
=��
=�aG��"=qC��q@
=���
���H��C��                                    Bx}`,�  �          @�=q@ff����B�\�
=C��@ff���H������\)C��\                                    Bx}`;V  
�          @��@*=q��  �n{�(��C��@*=q�x�ÿ������C�0�                                    Bx}`I�  �          @�ff@�R����aG��  C��@�R��  ��p���\)C���                                    Bx}`X�  �          @�Q�?�G����Ϳ
=��p�C�޸?�G������G���p�C�p�                                    Bx}`gH  "          @�=q@�H��{�(����(�C���@�H��z���
����C�n                                    Bx}`u�  
�          @�Q�@ff���
�(��أ�C�y�@ff���\��(����
C�<)                                    Bx}`��  �          @��@6ff��(��aG����C���@6ff��ff���\�aG�C�)                                    Bx}`�:  
�          @���@3�
����(�����C�@ @3�
���
�ٙ����HC�%                                    Bx}`��  �          @���@.{���ͿQ���
C��R@.{��=q������  C�ٚ                                    Bx}`��  �          @�G�@   ��Q�z�H�,��C�y�@   ������\����C���                                    Bx}`�,  
�          @�z�@5���������{C���@5���녿����33C��=                                    Bx}`��  �          @��H@8����ff�����C�y�@8������33����C�O\                                    Bx}`�x  "          @�(�@>{���R��\)�AG�C���@>{��Q쿫��g33C�k�                                    Bx}`�  T          @��\@<(���(��G���HC��\@<(���=q��ff��  C��                                    Bx}`��  �          @�=q@AG���녿(����C���@AG����ÿ�z����C�q�                                    Bx}aj  �          @���@E���
=�&ff���C�\@E��|�ͿУ���{C���                                    Bx}a  T          @�G�@L(���(��O\)���C�Ф@L(��u���G���C��                                    Bx}a%�  �          @�G�@L�����H�k�� z�C��@L���p�׿�{���HC�7
                                    Bx}a4\  
�          @�(�@=p���{�+����
C�� @=p������Q����\C��                                     Bx}aC  
�          @���@ ������+���G�C�B�@ ����=q��p���
=C�
=                                    Bx}aQ�  �          @�=q@(Q���녿333��\C�  @(Q����׿�  ����C��3                                    Bx}a`N  T          @���@7���z�Q��=qC��{@7����\������z�C��\                                    Bx}an�  �          @�  @)����\)�(����{C�Q�@)�����R��
=��p�C�!H                                    Bx}a}�  �          @���@*=q���׿@  ��RC�9�@*=q��
=���
���C�R                                    Bx}a�@  "          @���@����p��n{�"�HC���@�����\���R���C���                                    Bx}a��  �          @���@
�H���R�h���
=C�aH@
�H��(���(����
C�4{                                    Bx}a��  �          @�p�@C�
���H��
=��C�b�@C�
�x�ÿ����t(�C��                                    Bx}a�2  
Z          @�G�@e��c33�u�5C��H@e��\(��aG��   C�3                                    Bx}a��  
�          @�=q@j�H�_\)=#�
>�C�5�@j�H�Z=q�B�\�	�C��                                    Bx}a�~  
(          @��@n�R�c�
��\)�B�\C�,�@n�R�\�ͿaG���C���                                    Bx}a�$  
�          @�ff@XQ��|(�>��?ٙ�C�Q�@XQ��w��@  �Q�C���                                    Bx}a��  �          @�  @\(��|�ͼ��
�L��C���@\(��u�h��� ��C��                                    Bx}bp  T          @��H@Vff�u��L�Ϳ�C���@Vff�n{�h���%p�C���                                    Bx}b  �          @���@5��(��.{��Q�C�@ @5�~�R�����J�\C���                                    Bx}b�  T          @�@+��������
�n{C��H@+��z�H�}p��<(�C�(�                                    Bx}b-b  T          @�z�@"�\��=q�#�
���C��3@"�\�|(������M�C�ff                                    Bx}b<  �          @�ff@����?z�HA8Q�C�l�@���p��#�
���
C�!H                                    Bx}bJ�  
�          @�Q�@G����?c�
A$��C��@G���ff���Ϳ�
=C���                                    Bx}bYT  "          @�z�@(���=q?���As\)C���@(�����>�33@z�HC�4{                                    Bx}bg�  �          @�  @<(��w
=?�33ATQ�C��@<(�����>��@<��C�R                                    Bx}bv�  
�          @�z�@P���X��=#�
?�C��R@P���S�
�5�
�RC�J=                                    Bx}b�F  �          @���@qG��G
=����?�C�@ @qG��333�ٙ���
=C���                                    Bx}b��  
�          @���@|���A녿z����C�C�@|���4z῞�R�dQ�C�7
                                    Bx}b��  �          @�@p  �K��L�Ϳ\)C�޸@p  �E�@  ��C�E                                    Bx}b�8  
�          @�z�@l(��K�>�  @:=qC��H@l(��J=q�����
C���                                    Bx}b��  �          @�ff@n�R�4zᾞ�R�uC�p�@n�R�,(��n{�6{C��                                    Bx}b΄  �          @��\@R�\�>�R?^�RA1C��=@R�\�Fff>L��@!�C�Z�                                    Bx}b�*  �          @�
=@A��W
=@�\A�ffC�@A��n�R?�G�Ak�C���                                    Bx}b��  
�          @�{@>{�]p�?�A���C�H�@>{�q�?��AC�C�                                    Bx}b�v  
�          @��@Z�H�C�
=��
?uC�{@Z�H�@  �
=���
C�XR                                    Bx}c	  
�          @��@G��QG����
����C���@G��HQ쿂�\�O�C�}q                                    Bx}c�  T          @�ff@E�_\)>�z�@c�
C�˅@E�^{�����Q�C��                                     Bx}c&h  "          @�=q@N�R�\(�?fffA-p�C���@N�R�c�
>\)?ٙ�C�#�                                    Bx}c5  �          @�33@I���\��?�
=Ab�\C�:�@I���g�>��@���C���                                    Bx}cC�  T          @�z�@@  �aG�?�G�A��HC�:�@@  �qG�?:�HA��C�K�                                    Bx}cRZ  "          @���@>{�`��?�
=A�ffC�{@>{�r�\?c�
A'�C�                                    Bx}ca   
�          @��@<(��e?�ffA��C��=@<(��u?@  Ap�C���                                    Bx}co�  
�          @��\@>�R�g�?���AM�C���@>�R�p��>�\)@S�
C�,�                                    Bx}c~L  
�          @�
=@:�H�h��?�@θRC�XR@:�H�j�H��  �G
=C�<)                                    Bx}c��  "          @�33@L(��e�?��@���C��)@L(��hQ�.{���RC��                                    Bx}c��  �          @��\@@���dz�?�z�A_33C��@@���o\)>\@�G�C�j=                                    Bx}c�>  "          @��@'��dz�?��
A�
=C�!H@'��w
=?}p�A?33C��                                    Bx}c��  
�          @��@U��X��?:�HA\)C�@ @U��^{    <#�
C��                                    Bx}cǊ  �          @���@E��b�\?c�
A+\)C�� @E��i��>�?\C�{                                    Bx}c�0  "          @��@G
=�hQ�>��@�C�N@G
=�h�þ�33��\)C�G�                                    Bx}c��  
�          @�G�@Mp��`��?�@�33C�7
@Mp��b�\�W
=�#33C�{                                    Bx}c�|  T          @��@Tz��Z�H>�@���C��@Tz��\(�����EC�                                      Bx}d"  
�          @�Q�@L(��`��>Ǯ@��RC��@L(��`�׾�{����C��                                    Bx}d�  "          @�  @W��U�>��
@vffC��@W��U��\��G�C���                                    Bx}dn  "          @�\)@W��Q�>B�\@�\C��@W��P  ���H���C�
=                                    Bx}d.  �          @�@\(��Fff?�@�(�C���@\(��H�þ�����C�Ф                                    Bx}d<�  
�          @��R@Z=q�L��>��@���C�k�@Z=q�N�R�B�\�ffC�J=                                    Bx}dK`  �          @�ff@N�R�W�=���?�C��@N�R�Tz�����\C�)                                    Bx}dZ  �          @�p�@;��fff<#�
=�G�C��{@;��aG��:�H��
C��                                     Bx}dh�  
�          @�\)@9���mp�>L��@=qC��@9���j�H�\)�ڏ\C�&f                                    Bx}dwR  �          @���@P  �W�?W
=A"�\C���@P  �^{>\)?�z�C��\                                    Bx}d��  T          @�=q@X���O\)?��AG�C�"�@X���XQ�>�Q�@��C���                                    Bx}d��  
�          @�=q@X���AG�>aG�@5�C�%@X���@  �Ǯ��p�C�9�                                    Bx}d�D  �          @��@Vff�C33>���@r�\C�޸@Vff�C33���
����C��H                                    Bx}d��  �          @�p�@Mp��:=q?W
=A/�C���@Mp��AG�>u@FffC�b�                                    Bx}d��  �          @�p�@Vff�:�H?���A�{C�w
@Vff�H��?E�A�
C�p�                                    Bx}d�6  "          @��@\���333?˅A�p�C�q�@\���C�
?�  AC
=C�/\                                    Bx}d��  
�          @�G�@g��C33?�R@�C���@g��G
=    �#�
C���                                    Bx}d�  �          @��H@j�H�G�>�(�@��C�Ф@j�H�H�þB�\�33C���                                    Bx}d�(  �          @��R@b�\�C33>��@��C���@b�\�E��\)��\C�t{                                    Bx}e	�  �          @�\)@j�H�/\)?�Ae�C���@j�H�:�H?��@��HC��H                                    Bx}et  
�          @�Q�@k��1�?��A]C�xR@k��<��?�@ۅC��f                                    Bx}e'  T          @��@N�R�(�@z�A�ffC�}q@N�R�5?��A��HC�Q�                                    Bx}e5�  
�          @���@:=q�z�@J=qB&�C�O\@:=q�)��@,(�B	��C���                                    Bx}eDf  
�          @�z�@-p�����@n{BF��C��R@-p��33@U�B,z�C��q                                    Bx}eS  T          @��@ �׿�G�@���Bs�HC��@ ���z�@|��BW=qC��                                    Bx}ea�  �          @�@@���Fff?��A���C��@@���X��?�33Ac33C���                                    Bx}epX  �          @�
=@Dz��P  ?\A���C���@Dz��^�R?\(�A(��C��3                                    Bx}e~�  �          @�\)@>{�R�\>k�@=p�C�f@>{�Q녾�����HC��                                    Bx}e��  �          @�z�@��Tz�^�R�B�HC�Ff@��E���R��(�C�@                                     Bx}e�J  
�          @�z�@��S33������p�C��q@��@�׿����У�C��q                                    Bx}e��  T          @�ff@G
=�/\)�.{��
C�XR@G
=�)���0���p�C�Ǯ                                    Bx}e��  �          @���@.�R�<(����\�d��C�l�@.�R�,(�������{C��{                                    Bx}e�<  �          @�z�@1G��?\)�0�����C�b�@1G��3�
��  ��Q�C�N                                    Bx}e��  
�          @��
@A��C�
�@  �{C�\)@A��7
=�������RC�T{                                    Bx}e�  T          @��@8Q��K���p�����C��@8Q��C�
�xQ��O�
C��H                                    Bx}e�.  �          @�z�@@  �L�;aG��7�C���@@  �Fff�Q��-�C�f                                    Bx}f�  
�          @�Q�@;��C�
>�{@�G�C��)@;��Dzᾀ  �XQ�C���                                    Bx}fz  
�          @�G�@5��5���33����C�xR@5��-p��aG��J�HC�3                                    Bx}f    
�          @��\@(���<�Ϳh���P  C���@(���.{�������
C��                                    Bx}f.�  �          @���@&ff�Tz�\)��C�
=@&ff�O\)�B�\�%��C�ff                                    Bx}f=l  "          @�\)@?\)�fff?+�A�C��@?\)�j=q���
�uC���                                    Bx}fL  "          @�
=@E��_\)?8Q�A  C���@E��c�
=L��?(�C�p�                                    Bx}fZ�  
�          @���@@  �W�?�\)A^ffC���@@  �aG�>��@�Q�C�9�                                    Bx}fi^  "          @��@J�H�A�?��A���C�%@J�H�QG�?s33A<Q�C�{                                    Bx}fx  
Z          @�(�@L���;�?���A�(�C�@L���K�?��
AM��C���                                    Bx}f��  "          @��\@R�\�:�H?�G�A�C�5�@R�\�Fff?333A  C�W
                                    Bx}f�P  "          @�z�@hQ��0  ?G�A�HC�h�@hQ��6ff>�  @I��C��                                    Bx}f��  T          @�@c33�8Q�?�  AEG�C�w
@c33�AG�>�ff@�Q�C���                                    Bx}f��  "          @�{@`���7�?�
=Ah��C�b�@`���B�\?!G�@�
=C���                                    Bx}f�B  �          @�ff@Mp��Fff?�A�
=C��@Mp��S�
?Q�A"{C��                                    Bx}f��  �          @�@@  �N{?�=qA�Q�C�xR@@  �]p�?xQ�A>�\C�u�                                    Bx}fގ  �          @�{@=p��Q�?�ffA�z�C�@=p��`��?k�A5��C��                                    Bx}f�4  �          @�@���i��?��HA��C���@���y��?�G�AG�
C���                                    Bx}f��  �          @�{@��aG�?�(�AǙ�C��)@��tz�?�ffA�
=C��=                                    Bx}g
�  �          @�p�@&ff�H��@��A�33C���@&ff�^{?��
A�G�C�g�                                    Bx}g&  �          @�(�@Vff��H@z�AԸRC�R@Vff�0  ?˅A���C�K�                                    Bx}g'�  �          @�p�@_\)��\@z�A�33C�ff@_\)�'�?У�A�33C���                                    Bx}g6r  �          @�z�@^�R��@	��A�G�C��@^�R�"�\?�(�A�z�C��                                    Bx}gE  �          @�Q�@fff���R@z�A�C���@fff��?�
=A���C�K�                                    Bx}gS�  �          @�\)@vff�Q�?�Q�A��
C��@vff���?��
A{�C��                                    Bx}gbd  "          @�  @z=q�	��?˅A�G�C��@z=q�=q?�
=Ag33C�/\                                    Bx}gq
  T          @��@x���?�p�A���C�ٚ@x����?��AW�C��                                    Bx}g�  �          @�z�@�  �z�?��AS33C�J=@�  ��R?+�A�C�n                                    Bx}g�V  T          @�z�@dz����?�p�A�\)C�f@dz��*�H?��
A�C���                                    Bx}g��  T          @�
=@h���ff?�A�=qC���@h���(��?��A�C�                                      Bx}g��  "          @���@�=q��(�?޸RA�=qC��=@�=q�   ?�z�A�(�C��f                                    Bx}g�H  
�          @���@�(���ff?�A�ffC���@�(�����?�G�A�\)C��{                                    Bx}g��  "          @�=q@�G��
=q?\A�z�C��@�G����?�{AT��C��=                                    Bx}gה  
�          @���@hQ��*�H?\A�\)C��@hQ��9��?��
AG�C��                                    Bx}g�:  T          @��@_\)�.{?�Q�A��C��@_\)�>{?�
=Ag�C��=                                    Bx}g��  �          @�\)@fff�'�?�=qA�{C���@fff�6ff?��AV=qC�Ǯ                                    Bx}h�  �          @�
=@s33�#�
?���A]C�
=@s33�-p�?+�AC�<)                                    Bx}h,  �          @��@_\)�/\)?��A��C���@_\)�<(�?c�
A1G�C��                                    Bx}h �  �          @�z�@c�
�.{?�=qAY�C�K�@c�
�7�?
=@�C��\                                    Bx}h/x  �          @��
@g��0  ?8Q�A��C�aH@g��5�>k�@:�HC��
                                    Bx}h>  �          @�(�@o\)�%?^�RA,��C��@o\)�,��>��@���C�{                                    Bx}hL�  T          @�
=@\)��Q�?�\)A�(�C��@\)�(�?�G�Av�HC��H                                    Bx}h[j  
�          @��@z=q��p�?޸RA��HC��)@z=q�  ?�\)A�
=C�3                                    Bx}hj  
�          @�  @~�R�
=?�(�A���C��@~�R��?��AT(�C��{                                    Bx}hx�  �          @�@{��\)?��AS
=C�'�@{����?(��AffC�Y�                                    Bx}h�\  �          @�{@z=q�p�?(��A�RC��@z=q�"�\>k�@5C�~�                                    Bx}h�  �          @��R@u��(��?�R@�33C��R@u��-p�>#�
?�Q�C�aH                                    Bx}h��  �          @���@n�R�)��?   @�G�C�Q�@n�R�,(�=#�
?��C�{                                    Bx}h�N  
Z          @���@|(��{?��AO33C�Ff@|(���?&ff@��C�}q                                    Bx}h��  T          @��@����z�?�R@�
=C�s3@����	��>�  @K�C��                                    Bx}hК  T          @�  @��\�  ?aG�A+\)C���@��\��>��H@��
C���                                    Bx}h�@  �          @�
=@���	��?�\@ƸRC�Q�@���p�>\)?�C�                                    Bx}h��  
�          @��@�33�	��>���@���C�(�@�33���#�
��\C�f                                    Bx}h��  �          @�z�@��\��>\@�p�C���@��\�p�    =uC���                                    Bx}i2  �          @�(�@����p�>�G�@�  C���@������=u?J=qC�b�                                    Bx}i�  T          @��@���	��>�{@���C��@�������
����C��H                                    Bx}i(~  T          @��R@��Ϳ��;����j=qC�f@��Ϳ\�\)��z�C�q�                                    Bx}i7$  �          @�ff@�(����H�u�8Q�C�XR@�(�����{��ffC���                                    Bx}iE�  �          @��
@�Q��  �����C��\@�Q��(��������C��)                                    Bx}iTp  "          @��@�ff���
>�z�@g
=C��H@�ff���#�
�.{C�^�                                    Bx}ic  �          @P  @?\)���Ϳz�H���C�
@?\)=�\)�z�H��p�?�=q                                    Bx}iq�  "          @!G�?�\)?@  ��{�
=A�G�?�\)?xQ쿜(���  A��H                                    Bx}i�b  �          @Y���Ǯ@(��ٙ��
33B�Q�Ǯ@,(����\���HBĨ�                                    Bx}i�  �          @�ff�333@S33��  ���
CJ=�333@c33��
=�ip�C J=                                    Bx}i��  T          @���&ff@QG�������{C s3�&ff@a녿�  �
=B��q                                    Bx}i�T  "          @�p��'�@H�ÿ������RC�
�'�@W
=����_
=B��H                                    Bx}i��  
�          @��R�@  @*�H��=q��z�C
\)�@  @;���\)����C�f                                    Bx}iɠ  T          @�\)�Mp�@?\)��  ���\C	
=�Mp�@O\)���R�s�C                                    Bx}i�F  �          @�p��7
=@8���  ��G�C���7
=@N{�޸R��{C��                                    Bx}i��  �          @���O\)@3�
�
=q��  C��O\)@HQ������C�q                                    Bx}i��  
�          @�p��Z�H@>{��33���
C
=�Z�H@O\)������C�\                                    Bx}j8  T          @�p��K�@K�������C�K�@\�Ϳ�z���p�C�H                                    Bx}j�  �          @���.{@Mp���H��CT{�.{@c�
��\)��
=B���                                    Bx}j!�  
Z          @��
�:�H@HQ�����z�C�:�H@^{�����Q�C)                                    Bx}j0*  
�          @AG���?�p������̏\C33��@
=q�z�H���C�)                                    Bx}j>�  �          ?�\�!G�?\(����
�H�
B�  �!G�?�ff��\)�)B��                                    Bx}jMv  �          ?�
=�z�H?��ÿ����\)C���z�H?��R�p�����C u�                                    Bx}j\  "          ?��
�8Q�?�=q�Y�����B��
�8Q�?����!G���B�Ǯ                                    Bx}jj�  �          ?}p��u?@      >�B�
=�u?=p�=�A=qB�.                                    Bx}jyh  
�          ?�(����?����=p���ffB�
=���?�ff���H�k\)B�u�                                    Bx}j�  �          @a��
�H@ff��z���C��
�H@ �׿E��Q�C�                                    Bx}j��  �          @a��
=@!녿���(�C�=�
=@,(��@  �EG�C !H                                    Bx}j�Z  �          @�Q��,��@\)��(���33C	Q��,��@,(�����x(�C\                                    Bx}j�   "          @�
=�Mp�@�H��(����HC�R�Mp�@-p��Ǯ���HC޸                                    Bx}j¦  �          @����W
=@\)� ����Ck��W
=@1녿˅��=qC\)                                    Bx}j�L  "          @��H�Dz�@5����\)C	8R�Dz�@Fff��{��Q�C�                                     Bx}j��  T          @���A�@0�׿�p���ffC	�H�A�@B�\�\���C�)                                    Bx}j�  "          @��Fff@5��\��33C	z��Fff@HQ������ffC��                                    Bx}j�>  
�          @���Mp�@AG���33���RC�q�Mp�@R�\��33��p�CW
                                    Bx}k�  �          @��R�B�\@HQ��\��=qC��B�\@W���  �u��C�q                                    Bx}k�  �          @�  �A�@J=q��=q��p�C�R�A�@Z=q�����C��                                    Bx}k)0  �          @��X��@Dz����Q�C	���X��@S�
��ff�t��C�                                    Bx}k7�  T          @����@  @U���Q���33C  �@  @c33����\��C&f                                    Bx}kF|  �          @�z��7
=@g
=�У���  C T{�7
=@tz῅��D��B��                                     Bx}kU"  T          @�ff�;�@P  ����
=C���;�@^{����`��C)                                    Bx}kc�  �          @��H�7
=@E�����=qC�7
=@U���
��(�C�)                                    Bx}krn  
�          @�\)�-p�@.�R��p���33C���-p�@>{���
����Cp�                                    Bx}k�  �          @�33�S33@!녿�
=�Ǚ�C���S33@333��G���(�C��                                    Bx}k��  �          @�z��xQ�@&ff�����C(��xQ�@9����33����CB�                                    Bx}k�`  �          @��|��@,�Ϳ��H��(�C���|��@>�R��G���(�C                                      Bx}k�  �          @�p��x��@3�
��z����C+��x��@DzῸQ���C��                                    Bx}k��  T          @�p��u�@9�������  C�\�u�@J=q���~�HCk�                                    Bx}k�R  �          @��u@7
=���H��=qCE�u@HQ쿾�R��{C                                    Bx}k��  "          @�p��x��@8Q������p�C� �x��@HQ쿰���w\)C(�                                    Bx}k�  
(          @�p��u�@?\)��G����C�q�u�@N�R���
�d��Cٚ                                    Bx}k�D  �          @��R��ff@=q��33���C(���ff@*�H��G�����C�                                     Bx}l�  T          @�p��\)@0  ���
��33C^��\)@@  ��=q�n�HC{                                    Bx}l�  T          @�{��Q�@8�ÿ�������C5���Q�@Fff�����D(�CL�                                    Bx}l"6  �          @�ff���\@8�ÿ���w�
C�����\@Dz�k��#�
C�                                    Bx}l0�  �          @������@=p����
�ep�C������@G��L����C0�                                    Bx}l?�  �          @�(����@:=q���S�Ch����@C33�5���RC�                                    Bx}lN(  T          @�������@,(��s33�.=qC�����@333����  C�                                    Bx}l\�  �          @����{�@*�H�����b�\C�=�{�@4z�E��G�CL�                                    Bx}lkt  �          @�
=�z�H@2�\���R�g33C��z�H@<�ͿL����C                                    Bx}lz  
�          @�Q��\)@3�
���
�=�C�{�\)@<(��z���ffC��                                    Bx}l��  �          @�����@0  �Y�����C�H���@6ff��
=��G�C�3                                    Bx}l�f  �          @�{�n{@N{��=q���RC&f�n{@Z�H�����?�C	c�                                    Bx}l�  T          @�����p�@<�Ϳ��H�U��C����p�@Fff�@  ��HCY�                                    Bx}l��  T          @������
@;�������C&f���
@>�R���
�W
=C��                                    Bx}l�X  T          @������
@>{�����\)C�����
@<��>��R@Tz�C��                                    Bx}l��  �          @�Q���\)@@  �aG��{C�f��\)@Fff������C                                    Bx}l�  �          @�����@E�����>{C�����@N{�
=�ϮC�
                                    Bx}l�J  T          @����|(�@7���(��aG�C��|(�@AG��E��p�C�                                    Bx}l��  �          @�ff���@+��h���)�C�����@2�\���H��
=C��                                    Bx}m�  �          @���~�R@3�
���
�>�HC�~�R@<(��
=���C�
                                    Bx}m<  
�          @����~{@8�ÿc�
�#�C��~{@?\)��G�����C�q                                    Bx}m)�  �          @����u�@(��Q���33Cp��u�@/\)�޸R��  Ch�                                    Bx}m8�  
�          @�\)�hQ�@$z��  ��ffC�q�hQ�@8Q��=q��z�C��                                    Bx}mG.  T          @�ff�e@33�\)���CT{�e@*=q�
=��Q�C�=                                    Bx}mU�  �          @�
=�l��@Q���\���
C5��l��@,�Ϳ�33��ffC�)                                    Bx}mdz  �          @�G��c33@E�޸R��ffC
�3�c33@Tzῠ  �g�C�                                    Bx}ms   
�          @���\��@R�\�������C\)�\��@a녿�=q�r{CQ�                                    Bx}m��  T          @����`  @N�R��\)���
C	B��`  @\(���{�K�
C}q                                    Bx}m�l  4          @�{�g�@@�׿�p���G�CE�g�@L�Ϳ�G��<  C
��                                    Bx}m�  �          @��R�Z�H@H�ÿ�
=��\)C	n�Z�H@W
=��Q��_
=C��                                    Bx}m��  �          @�ff�W�@Dz����G�C	���W�@TzῸQ����Cff                                    Bx}m�^  �          @��
�b�\@A녿����Y��Cn�b�\@J=q�(����z�C
0�                                    Bx}m�  T          @��H��Q�?�����R����C���Q�@z�ٙ�����C�3                                    Bx}m٪  �          @�33����?�������C
����?�
=��\)���C                                    Bx}m�P  �          @�z����R?����33����C"8R���R?�녿�����C��                                    Bx}m��  �          @������R?��R�   ���RC }q���R?�\��  ���C#�                                    Bx}n�  �          @�33��Q�?�ff� ����ffC�=��Q�@���(���=qC�=                                    Bx}nB  �          @��
�|��@녿�p���Cٚ�|��@33��33����C��                                    Bx}n"�  �          @�G��x��?�  ����=qCǮ�x��@�
������C#�                                    Bx}n1�  �          @�=q�dz�?�Q��5��ffC��dz�?����%����C��                                    Bx}n@4  �          @�
=�fff@�
�ff��\CT{�fff@(Q��p����C��                                    Bx}nN�  �          @�ff�e@���\��z�C�\�e@,(���z�����C0�                                    Bx}n]�  �          @�ff�u�?��R�����p�C�\�u�@���
�ŮC                                    Bx}nl&  �          @�{�q�@ �������Q�C�q�@ff�����C)                                    Bx}nz�  �          @��R�s�
@  �{��
=Cff�s�
@#�
��{��Q�C)                                    Bx}n�r  �          @��p  @��\)��  C���p  @%��\)��z�CY�                                    Bx}n�  �          @�z��z=q@   �����  C�R�z=q@�\�������HC��                                    Bx}n��  
�          @����~{@\)��{����C�H�~{@-p���(��c�
C�q                                    Bx}n�d  �          @�{�n�R@�
��R��C��n�R@=q�	����z�C.                                    Bx}n�
  �          @�\)�Q�@'
=�!���33Cz��Q�@=p����̸RC	�                                    Bx}nҰ  �          @��R�S33@%��(��� ��C�q�S33@<(��{�ՅC
E                                    Bx}n�V  �          @�ff�]p�@%�����G�C5��]p�@:�H��p����\C��                                    Bx}n��  �          @���`��@�R���߮C� �`��@2�\�����ffC}q                                    Bx}n��  �          @���]p�@ �������\)C��]p�@5���R��p�C�H                                    Bx}oH  �          @����Z=q@>�R�Q��ɮC
���Z=q@P�׿���p�CO\                                    Bx}o�  �          @�G��7�@_\)�p��У�Cu��7�@q녿�����B�u�                                    Bx}o*�  �          @����>{@N{�(���G�C�R�>{@c33��Q���C��                                    Bx}o9:  �          @�
=�=p�@G��!G���RCp��=p�@]p��G����C�                                     Bx}oG�  �          @����7�@5��'
=���CaH�7�@L(��
=q��Q�C��                                    Bx}oV�  T          @��\�G�@3�
���뙚C
�G�@HQ��
=��ffC�                                    Bx}oe,  �          @��H��@0���z��ҏ\C
���H��@B�\�����{C��                                    Bx}os�  �          @�=q�W
=@0����
��33C�
�W
=@B�\�У���ffC	�H                                    Bx}o�x  �          @���\��@(���  ����C���\��@<�Ϳ�=q���C}q                                    Bx}o�  �          @�z��c33@)���33����C:��c33@;��У���ffCz�                                    Bx}o��  �          @��
�g�@(Q��
=���RC��g�@8�ÿ\��
=C\)                                    Bx}o�j  �          @����e@)���޸R��=qC���e@8Q쿪=q��z�CO\                                    Bx}o�  �          @�\)�^�R@*�H��\��\)Cs3�^�R@:=q������ffC�                                    Bx}o˶  �          @�\)�`  @4zΌ�����HC&f�`  @@  ��G��F{C^�                                    Bx}o�\  T          @�33�dz�@:=q��(����C�)�dz�@Fff���\�C33C)                                    Bx}o�  �          @��\�fff@8Q쿸Q���\)CY��fff@C�
��  �?\)C��                                    Bx}o��  �          @���dz�@=p����H��  C^��dz�@H�ÿ�G��?
=C
��                                    Bx}pN  �          @����`  @Dz���
��p�C
� �`  @P�׿�ff�F�HC	�                                    Bx}p�  �          @�p��Z=q@Fff��p���(�C	�f�Z=q@U����R�j=qC�                                    Bx}p#�  �          @�
=�W
=@N�R�ٙ����\C0��W
=@\(������`z�CQ�                                    Bx}p2@  �          @�(��N{@N{��p���Q�C
=�N{@\(���p��i�C!H                                    Bx}p@�  �          @�p��J=q@K����H���HC�
�J=q@\(���(���G�C��                                    Bx}pO�  �          @���J=q@B�\�33�ȏ\C!H�J=q@S�
��=q���C��                                    Bx}p^2  �          @X�ÿ�{@ff��\��B��Ϳ�{@Q�ٙ���G�B�                                      Bx}pl�  �          @W���
=?�\)�(��5Q�Cs3��
=?�
=�����B�#�                                    Bx}p{~  �          @XQ��\@ff�˅��p�B�녿�\@#�
��(����B�=q                                    Bx}p�$  �          @U��@녿���	z�C.��@�\��ff�ޣ�C �\                                    Bx}p��  �          @U��G�@녿ٙ���{B�B���G�@ �׿������B�\                                    Bx}p�p  �          @0  >�
=�!G���33��C���>�
=�\��p���C�=q                                    Bx}p�  �          @%�?fff��  �\(���
=C��q?fff���Ϳ�{���C���                                    Bx}pļ  �          @*=q����?.{�z��Y�C�q����?z�H�����G=qCW
                                    Bx}p�b  T          @dz��\)?Q��z��&=qC!޸�\)?�33����\C#�                                    Bx}p�  �          @����G
=?����7��#��C ���G
=?\�*�H���C�                                    Bx}p�  �          @�p��H��?��H�>�R�z�CY��H��@���,(��p�C�                                    Bx}p�T  �          @��R�<(�@�\�7��G�C��<(�@,(��\)�   C	�                                     Bx}q�  �          @����4z�@+��#33�{C���4z�@A������C�q                                    Bx}q�  �          @����%�@{�<���#�RCW
�%�@(Q��%���Ch�                                    Bx}q+F  �          @���7
=@���#�
��RCp��7
=@3�
�
=q����C��                                    Bx}q9�  �          @����7
=@4z��ff��=qC^��7
=@H�ÿ�33��Q�CJ=                                    Bx}qH�  �          @�
=�G
=@@  ������C�G
=@P  ����{C��                                    Bx}qW8  �          @�{�1�@C33�������CT{�1�@Vff��(����\C�3                                    Bx}qe�  T          @��A�@>�R������33C��A�@O\)��p����C!H                                    Bx}qt�  �          @�p��A�@C�
��ff��33C��A�@S33������
=C�\                                    Bx}q�*  �          @�Q��AG�@333��\��=qC	@ �AG�@A녿�=q��(�C�                                    Bx}q��  �          @�z��(��@HQ������
=C
�(��@U��{�jffC L�                                    Bx}q�v  �          @����>{@H�ÿ����Cff�>{@Tz�n{�<  C�{                                    Bx}q�  �          @��R�A�@G���G��P��C(��A�@N�R���أ�C�                                    Bx}q��  �          @��E�@W
=�z�H�B�\C� �E�@^{�����RC�{                                    Bx}q�h  �          @�=q�,��@J=q������=qC}q�,��@Y��������z�C ff                                    Bx}q�  �          @�p��&ff@J�H���R��=qCL��&ff@W
=��  �S\)B�\)                                    Bx}q�  �          @�p��\)@`�׿B�\�!G�B��\)@e��aG��<��B���                                    Bx}q�Z  �          @�ff�
=@c�
��p�����B���
=@mp��.{�z�B��)                                    Bx}r   �          @�33�\)@Q녿�  ����B����\)@`  ���R��z�B�\                                    Bx}r�  T          @�Q���@%�{��\CJ=��@<(��33��33B��\                                    Bx}r$L  �          @�33�"�\@.{�p����C{�"�\@A녿�\��\)C�                                    Bx}r2�  �          @����H@>�R��
��\)C �q��H@P�׿�=q��{B�\                                    Bx}rA�  T          @�(��   @=p��33���C#��   @O\)��=q���\B�B�                                    Bx}rP>  �          @��R�\)@1G���33��=qC�R�\)@AG����H��
=Cu�                                    Bx}r^�  �          @�=q�ff@dzῦff����B�
=�ff@n{�=p��Q�B��                                    Bx}rm�  �          @����z�@l(��:�H��\B���z�@p�׾���ffB��                                    Bx}r|0  �          @���(�@^{��\)��(�B��{�(�@h�ÿTz��0��B�#�                                    Bx}r��  �          @�{��H@Z�H��������B����H@g
=�h���=��B�                                    Bx}r�|  �          @�{�,��@K���G�����CE�,��@XQ쿀  �P��C �)                                    Bx}r�"  �          @�\)��(�@|�Ϳ��H�{�B�  ��(�@��H�
=��B�{                                    Bx}r��  �          @���@u�}p��Lz�B��@|�;Ǯ��  B�R                                    Bx}r�n  T          @��ÿ�=q@y�������
=B�aH��=q@����+��	B�R                                    Bx}r�  �          @��\���
@�  ��ff�UG�B՞����
@���Ǯ��33B���                                    Bx}r�  T          @�녿G�@�Q쿨�����B�Ǯ�G�@���&ff�z�B�\                                    Bx}r�`  �          @����R@HQ�xQ��i�B�녿��R@O\)�����33B��                                    Bx}s   �          @�\)���?��ÿ�(����C"8R���?��ÿ�  ��Q�C�                                    Bx}s�  �          @����Tz�@K������33C@ �Tz�@Y�������V=qCaH                                    Bx}sR  �          @�ff�a�@B�\����  CE�a�@QG���
=�]�C	B�                                    Bx}s+�  �          @����Vff@4z����33C�3�Vff@E����H��{C	h�                                    Bx}s:�  �          @���P  @5������¸RC
���P  @Fff���R���\Cff                                    Bx}sID  �          @��H�u�?��R�p����C���u�?��ÿ����ř�C�)                                    Bx}sW�  �          @�=q��
=?\�ff����C @ ��
=?�{����Q�C33                                    Bx}sf�  �          @�33����?����\��
=C������@{���H��G�C�
                                    Bx}su6  �          @�(���@z��������C����@Q�����G�CY�                                    Bx}s��  T          @��H��p�@	��������z�C���p�@��˅���
C�                                     Bx}s��  �          @������@����ffC������@&ff�������RC#�                                    Bx}s�(  
�          @�����(�@���\)��{CL���(�@ff�\��  Ck�                                    Bx}s��  �          @������H?��H��{��(�CxR���H?�p��������
Cp�                                    Bx}s�t  �          @�\)�}p�@���z�����CxR�}p�@'
=�\��z�C��                                    Bx}s�  �          @������@�����H��{C�����@(�ÿ���s
=C�                                    Bx}s��  �          @����w�@6ff��(���(�C��w�@B�\��G��8z�C�
                                    Bx}s�f  �          @����r�\@:�H��=q��  Ch��r�\@HQ쿌���Ip�Cs3                                    Bx}s�  �          @��H�U@U�������C�q�U@e��ff�mp�C�                                    Bx}t�  �          @�Q��U@X�ÿ�  ��{C���U@e�s33�/
=C                                      Bx}tX  �          @�  �h��@G
=��z����HC���h��@R�\�fff�%p�C	�                                    Bx}t$�  �          @��\�q�@K���z��R�HC޸�q�@U��#�
��
=C
�H                                    Bx}t3�  �          @������@Vff�k�� ��CxR���@Vff>���@S�
C�                                    Bx}tBJ  �          @�Q���=q@Tz�=�G�?�Q�C�\��=q@P��?!G�@�ffCO\                                    Bx}tP�  T          @��
�l��@J�H?uA2�\Cff�l��@>�R?���A�(�C.                                    Bx}t_�  T          @�33�X��@n�R����
=C=q�X��@qG�=��
?Y��C��                                    Bx}tn<  �          @�33�j�H@a녾���9��C��j�H@a�>��R@_\)C!H                                    Bx}t|�  �          @�(��j=q@c�
���
�uC�
�j=q@aG�?��@�p�C.                                    Bx}t��  �          @����fff@]p�>#�
?�C.�fff@X��?5A�RC�                                    Bx}t�.  �          @����W�@_\)>�{@��C
=�W�@X��?fffA*�RC�H                                    Bx}t��  �          @������@�녿���J=qB�=q���@��\����B��f                                    Bx}t�z  �          @�33���@�ff��G��?�B�L����@�녾��
�n�RB��                                    Bx}t�   �          @���33@�(��Q����B�8R�33@�ff���Ϳ�B�z�                                    Bx}t��  �          @����
=@��R�.{� Q�B��Ϳ�
=@�Q�=L��?�B�Q�                                    Bx}t�l  �          @�p���p�@��H�����Q�B�ff��p�@��H>�{@}p�B�W
                                    Bx}t�  �          @�
=�33@�\)�
=q��  B�33�33@�Q�>L��@��B��                                    Bx}u �  �          @��Q�@�=q�z�H�8��B�p��Q�@�p���=q�G�B�aH                                    Bx}u^  �          @�����R@�?Tz�A(��B�.���R@\)?�  A���B��f                                    Bx}u  �          @�z���@��R?8Q�A�B߸R���@�G�?�Q�A�p�B�\)                                    Bx}u,�  �          @����Q�@��R?!G�@�(�B����Q�@�G�?���A�
B�\                                    Bx}u;P  �          @�����\@��>��?�G�B�q��\@�
=?Y��A$Q�B癚                                    Bx}uI�  �          @��H��@�(�������B���@��H?��@�B�.                                    Bx}uX�  �          @��H�\)@�Q쾮{����B�u��\)@�  >�Q�@��\B�z�                                    Bx}ugB  �          @���G�@��H���
�k�B�  ��G�@���?=p�A	Bހ                                     Bx}uu�  �          @����(�@�(��������HB����(�@�(�>���@�  B��                                    Bx}u��  T          @�녿�ff@��\�����ffB�p���ff@��
>.{@
�HB�33                                    Bx}u�4  �          @��
���
@�33��ff����B��Ϳ��
@��>��
@qG�BظR                                    Bx}u��  �          @�(���p�@��׿(���z�B��Ϳ�p�@��>��?�ffB�z�                                    Bx}u��  �          @����\@��
�z����B�����\@��>#�
?��B垸                                    Bx}u�&  �          @������@�p�����Q�B�{����@�{>W
=@!�B�Q�                                    Bx}u��  T          @�녿��@����=q�QG�B�����@�z�>��@�=qB�B�                                    Bx}u�r  �          @��\��ff@`���   ��
B��쿦ff@w���\)��
=B�8R                                    Bx}u�  �          @�=q��z�@x�ÿ��ƸRB���z�@�����R�{�B�ff                                    Bx}u��  �          @�G���  @mp��Q���z�B����  @��׿�p���Q�B�\                                    Bx}vd  �          @�녿��@��R>��@I��B�\���@�33?�G�AD��B��                                    Bx}v
  �          @���У�@�(�?�  Ao�B��H�У�@�33?�(�A�Q�B�\)                                    Bx}v%�  �          @�
=���@��?fffA0z�B�Ǯ���@���?�{A��B�                                    Bx}v4V  T          @��R���@���>�ff@��RB㙚���@���?�AeB�                                      Bx}vB�  �          @��H��ff@���?(�@��B���ff@~�R?��A�p�B��                                    Bx}vQ�  �          @��@  @U�@��B{B�k��@  @7�@:�HB2�B�L�                                    Bx}v`H  T          @���p��@e�?�G�A�{Bр �p��@N{@ffB�HBԏ\                                    Bx}vn�  T          @��\��ff@�{>�
=@��B� ��ff@���?���Ad(�B��
                                    Bx}v}�  T          @��
��Q�@�  �J=q�!�B����Q�@�=q���Ϳ�p�B���                                    Bx}v�:  T          @����   @����z�H�C33B虚�   @����=q�Tz�B�k�                                    Bx}v��  �          @��
�33@>�R�+���
B�=q�33@Y���Q���\)B��                                    Bx}v��  �          @��H��
@9���(Q��z�C �\��
@S�
�ff��G�B��)                                    Bx}v�,  �          @�����@7
=�)���C.��@QG��
=��(�B��f                                    Bx}v��  �          @��R�33@'
=�<���&p�C 33�33@E������B�aH                                    Bx}v�x  �          @�  ��p�@   �J=q�2�HC Y���p�@@���+��p�B���                                    Bx}v�  �          @�\)�˅@1G��HQ��1�RB���˅@QG��&ff�G�B��                                    Bx}v��  �          @�\)��@G�����HB�  ��@^�R��G����B�=q                                    Bx}wj  �          @�\)��\@S�
��z��ʸRB�k���\@e��ff���B��                                    Bx}w  T          @�Q��=q@U��  ����B��3�=q@e����m�B��R                                    Bx}w�  �          @���
=@P������Q�B�Ǯ�
=@e���G���B��q                                    Bx}w-\  �          @�ff�	��@Tz��"�\���B�Ǯ�	��@n{�����RB�
=                                    Bx}w<  �          @��R�
=@e����ޏ\B�  �
=@z�H���
��B�                                    Bx}wJ�  �          @����33@n{��
=���HB��33@�  ��  �q�B��f                                    Bx}wYN  �          @��\�=q@o\)��33���HB��{�=q@�Q쿙���g
=B��                                    Bx}wg�  �          @�����@n{��Q���(�B��
���@�Q쿠  �n=qB�                                      Bx}wv�  �          @�z��%@mp������B����%@~{��33�Z{B��                                    Bx}w�@  �          @����,��@u������=qB�B��,��@��׿#�
���B��
                                    Bx}w��  �          @�Q��,(�@�
==#�
>�B���,(�@�z�?J=qA�B�\                                    Bx}w��  �          @��H�ff@�ff��z��_\)B�W
�ff@�{>�@���B�                                    Bx}w�2  �          @����(�@z�H��������B�Q��(�@{�>��
@��B�=q                                    Bx}w��  �          @�녿��@s33�z�����B��)���@vff=�G�?�  B�\)                                    Bx}w�~  �          @����@c�
��
=����B�u���@s�
��G��L��B�                                      Bx}w�$  �          @�ff�!G�@c�
������B��\�!G�@s33��  �Ep�B�                                    Bx}w��  �          @�p����@o\)�����^{B�ff���@xQ������HB�                                     Bx}w�p  �          @�
=��@\)�L���G�B���@�=q��\)�L��B��                                    Bx}x	  �          @��R�%@w��\)��z�B����%@z=q>��?���B�{                                    Bx}x�  �          @�
=�'�@y������G�B��H�'�@z�H>u@<��B��{                                    Bx}x&b  �          @���=p�@l(�=#�
?�C ��=p�@g�?8Q�A�
C8R                                    Bx}x5  �          @�G��9��@q�=���?�Q�B���9��@l��?J=qA��C �                                    Bx}xC�  �          @�Q���
@�(����H���B�G����
@���>��
@z�HB�#�                                    Bx}xRT  �          @�����@�
=���
�w
=BꞸ��@�ff>�@��\B�Ǯ                                    Bx}x`�  �          @����{@��?�ffAL��B�G��{@s33?��
A��\B�{                                    Bx}xo�  �          @�33�(�@w�?�{A�\)B��(�@`  @33A��B�Ǯ                                    Bx}x~F  �          @����@k�@   A�
=B�����@O\)@(��B
=B��                                    Bx}x��  �          @�
=��p�@tz�?�Q�Axz�B���p�@a�?��A��B�=                                    Bx}x��  �          @�z��(�@��\?�Al(�B���(�@r�\?�z�A���B��)                                    Bx}x�8  T          @�=q���H@�p�?8Q�A
=B�aH���H@�ff?��A�B�W
                                    Bx}x��  �          @�  ��33@����L�Ϳ��B�R��33@��?B�\A��B�ff                                    Bx}xǄ  �          @�G��6ff@ff�&ff�z�Cz��6ff@333�
=�޸RC��                                    Bx}x�*  �          @��\�:=q@�
�)���
=C�\�:=q@1G��
�H���HCn                                    Bx}x��  �          @�z��A�?޸R�@���#
=C&f�A�@��(Q��
�C{                                    Bx}x�v  �          @�Q��A�@C�
��������CǮ�A�@Vff��p��t  C&f                                    Bx}y  �          @�p��>�R@p�׿�G��>�RC aH�>�R@x�þ�\)�QG�B���                                    Bx}y�  �          @�z��5�@vff�xQ��6�RB��\�5�@~{�W
=� ��B���                                    Bx}yh  �          @��>{@j�H������C��>{@w��+���G�B��                                    Bx}y.  �          @�ff�E@j=q���H�bffC5��E@tz����(�C �R                                    Bx}y<�  �          @�{�E@\(����H��{C�R�E@l�Ϳ�G��>{C�H                                    Bx}yKZ  �          @�(��3�
@dz����{C G��3�
@vff����NffB�\)                                    Bx}yZ   �          @��_\)?�(��O\)�%  C ��_\)?�=q�<���(�C^�                                    Bx}yh�  T          @��R�Y��?�{�Mp��!��C��Y��@���5�(�C�                                    Bx}ywL  �          @�Q��J�H?��\�XQ��5��C":��J�H?�33�G��$G�CxR                                    Bx}y��  �          @�  �E�#�
�e�D�HC6��E?
=q�c33�B{C*
=                                    Bx}y��  �          @�z��L(�@A녿�z���\)CxR�L(�@Vff���
�w\)C�f                                    Bx}y�>  �          @�  �N�R@/\)�(Q���Q�C���N�R@L����\��
=C8R                                    Bx}y��  �          @�ff�J=q@-p��+��
=CW
�J=q@L(����=qC�                                     Bx}y��  �          @�
=�Dz�@2�\�.{��C	�3�Dz�@QG��
=��  C(�                                    Bx}y�0  �          @�ff�<(�@0���7���RC���<(�@QG��G���=qC��                                    Bx}y��  �          @�ff�;�@(Q��>�R�ffC
{�;�@J�H�����\C                                    Bx}y�|  �          @�  �>�R@0  �8Q��z�C	G��>�R@QG������C\)                                    Bx}y�"  �          @�Q��U�@U��Q���p�Cٚ�U�@g
=�}p��6=qC��                                    Bx}z	�  �          @����Tz�@^�R��Q���\)C�f�Tz�@l�Ϳ8Q���C�                                    Bx}zn  �          @��H�dz�@`  �u�-p�C���dz�@g��u�,(�C�H                                    Bx}z'  �          @����g�@P�׿��
�j�HC	�R�g�@\�Ϳ�����C\)                                    Bx}z5�  �          @�  �Vff@L�Ϳ�ff��(�CT{�Vff@`  ��\)�Q�CǮ                                    Bx}zD`  �          @���Mp�@Tz��=q��p�C\�Mp�@g������R=qC��                                    Bx}zS  �          @���U@C33��z����RC	���U@W���  �l(�CǮ                                    Bx}za�  �          @����]p�@N{�޸R��C	��]p�@`�׿�ff�@��C�{                                    Bx}zpR  �          @����Fff@s33����Hz�C:��Fff@|(������XQ�C 5�                                    Bx}z~�  �          @��
�K�@G
=���H���
C�H�K�@\(����
�tz�C�R                                    Bx}z��  �          @�{�\��@C�
��{���C
u��\��@Tz�s33�333C�                                    Bx}z�D  �          @�p��mp�@p���
�ƸRCff�mp�@5���G����C��                                    Bx}z��  �          @�{�qG�@
=������C�\�qG�@#33�����RC�H                                    Bx}z��  �          @���fff?��R�)���33C!H�fff@\)�(���(�CY�                                    Bx}z�6  �          @�p��o\)?�G��*�H�Q�C޸�o\)@G��  ��p�CǮ                                    Bx}z��  �          @�{�j=q?\�<(��\)C���j=q@�#�
��z�CB�                                    Bx}z�  �          @�ff�u�?��
�<����\C%  �u�?�\)�*�H��C�                                    Bx}z�(  �          @�z��j�H?E��HQ����C(33�j�H?�33�9���
=C
                                    Bx}{�  �          @�Q��c�
>���G��$�
C/�f�c�
?fff�@  �{C%�=                                    Bx}{t  �          @�\)�dz�>aG��Dz��"C0z��dz�?\(��=p���\C&z�                                    Bx}{   �          @���[���\)�HQ��)C5+��[�?
=�E��&�C*@                                     Bx}{.�  T          @�(��Mp��L���S�
�7p�C7� �Mp�?�\�Q��5�C+\                                    Bx}{=f  �          @�33�J=q��{�L(��4��C:
�J=q>�{�L(��4��C-Ǯ                                    Bx}{L  �          @�p��6ff?0���O\)�?33C&c��6ff?�{�AG��.�RC�                                    Bx}{Z�  �          @�G��G�@B�\���R��33B�\�G�@Y����ff��{B�aH                                    Bx}{iX  �          @����G�@?\)�)����B�(��G�@^�R������z�B�G�                                    Bx}{w�  �          @����H@W
=�33��  B��\���H@n{���
��B랸                                    Bx}{��  �          @��\��p�@`���	�����\B��Ϳ�p�@x�ÿ�����\)B��                                    Bx}{�J  �          @��H��z�@c33��R��B왚��z�@|(�����\)B�                                    Bx}{��  �          @�
=��z�@X���{��\)B�q��z�@r�\��Q�����B�z�                                    Bx}{��  �          @�ff�@XQ��   ��\)B�z��@n�R��(��~�RB�                                    Bx}{�<  �          @�(���Q�@O\)�����{B�q��Q�@h�ÿ�������B�
=                                    Bx}{��  �          @�zῪ=q@fff�Q���z�B�zῪ=q@~�R��ff���B��                                    Bx}{ވ  �          @�(���
=@R�\�z�� z�B����
=@mp���ff��Q�B��                                    Bx}{�.  �          @��Ϳ�
=@Tz���
���HB����
=@o\)���
���
B�Q�                                    Bx}{��  �          @�z���\@L���*=q�=qB���\@l�Ϳ�33��p�B���                                    Bx}|
z  �          @�=q��@G��@  �
=B����@l���  ��{B�3                                    Bx}|   �          @�녿���@?\)�N{�(33B������@g���R����B�z�                                    Bx}|'�  �          @����@:�H�N�R�)�B�����@c�
� �����
B�\)                                    Bx}|6l  �          @�=q�{@?\)�C33�G�B�#��{@e��
��B�k�                                    Bx}|E  �          @�=q��@!G��W
=�0��C:���@L���-p��	(�B�                                      Bx}|S�  �          @�G��ff@(���N�R�)��C�R�ff@R�\�$z���B��                                    Bx}|b^  �          @����(Q�@z��N{�*33C
�{�(Q�@>�R�'����Cn                                    Bx}|q  �          @����,(�@ff�@  �%33C\�,(�@.{�p��C�q                                    Bx}|�  �          @�=q�+�?�=q�Vff�<p�C}q�+�@�\�:=q��C}q                                    Bx}|�P  �          @�(�� ��@&ff� ���ffB�aH� ��@Fff��{���
B��                                    Bx}|��  �          @����˅@Z�H��������B��Ϳ˅@l(��B�\�+33B♚                                    Bx}|��  �          @����?�Q��`  �F
=C����@+��>{� �HCaH                                    Bx}|�B  �          @���@   �XQ��D��C�H��@.{�5��
B��                                    Bx}|��  �          @�\)���@{�N{�8��C�f���@8���(Q��z�B��f                                    Bx}|׎  �          @��H�33@��Tz��:C	\)�33@4z��0  �33C8R                                    Bx}|�4  �          @���?�Q��e��Q�
CW
�@{�I���1Q�Cz�                                    Bx}|��  T          @�=q�!�@�R�C�
�(�RC
�
�!�@8Q��{�{C^�                                    Bx}}�  �          @�
=����?�(��j=q�^p�C������@!G��J�H�6G�B��f                                    Bx}}&  �          @��׿�ff?n{�u��\C���ff?���aG��`\)C��                                    Bx}} �  �          @�(����>���z�H��C'.���?�z��p  \C��                                    Bx}}/r  �          @�Q쿨��>�G��n�R{C!�q����?���a��yz�C�R                                    Bx}}>  �          @�ff��(�?n{�tz�  C�\��(�?���`  �a�Cs3                                    Bx}}L�  �          @������?�\)�h���`�C
Y�����@�H�J=q�8��B���                                    Bx}}[d  �          @��H��\?�p��Z=q�A�\C5���\@.{�6ff��HC�                                    Bx}}j
  �          @����0��@��(����CaH�0��@5����33C.                                    Bx}}x�  �          @�녿�
=@G��aG��[��B����
=@2�\�<(��,�B�33                                    Bx}}�V  �          @�G���p�?�\�j=q�f��C�)��p�@%�H���9B                                    Bx}}��  �          @�����@/\)�l���=\)B�{��@b�\�<���p�B�R                                    Bx}}��  �          @�����@.�R�j=q�:��B��\��@aG��:=q�33B�                                    Bx}}�H  �          @��R��
@5��^{�3{B�����
@e��,�����B���                                    Bx}}��  �          @�z��'�@  �Fff�'��CQ��'�@;���R� �HC�
                                    Bx}}Д  �          @�\)�
�H@A��8Q���RB��
�H@hQ��z��Џ\B�\                                    Bx}}�:  �          @��
���R@AG��7���\B��
���R@g���
���B��                                    Bx}}��  �          @�=q��Q�@#�
�Fff�5(�B��)��Q�@N�R�����B�=q                                    Bx}}��  �          @�ff����?����z=q�a�Ck�����@-p��W
=�6�\B���                                    Bx}~,  �          @�Q���@��L(��:�C �����@Dz��!G��(�B�{                                    Bx}~�  �          @�=q�ٙ�@r�\��{��\)B�W
�ٙ�@�(��c�
�4z�B�Ǯ                                    Bx}~(x  �          @��H���
@n�R����\)B�����
@��Ϳ��H�t��B܊=                                    Bx}~7  �          @�{��(�@Z�H����{B�k���(�@y�����R��\)B�8R                                    Bx}~E�  �          @����@W��>�R�Q�B�Q��@\)�z���\)B�z�                                    Bx}~Tj  �          @�Q��33@Tz��G��B�G��33@~�R�p���z�B�\)                                    Bx}~c  �          @���G�@Tz��H�����B���G�@\)��R���HB�=q                                    Bx}~q�  �          @�\)��@XQ��L���=qB����@�=q�G���(�B�aH                                    Bx}~�\  �          @�33����@�
=��\��
=B噚����@�33�n{�(��B�                                    Bx}~�  �          @�녾�z�@�
=�   ����B�Ǯ��z�@�{?=p�A��B���                                    Bx}~��  �          @��þ��@~{@"�\B��B�� ���@Mp�@\(�B:�RBĞ�                                    Bx}~�N  �          @�ff�}p�@/\)@A�B8\)Bۙ��}p�?�33@g
=BmffB��H                                    Bx}~��  �          @�p��c�
@��@G�BHp�B���c�
?���@h��B}G�B�8R                                    Bx}~ɚ  �          @n{>�
=@A�?�z�A�B�
=>�
=@!G�@
=B+33B��
                                    Bx}~�@  �          @Y��?��@J=q=�\)?�\)B���?��@B�\?\(�Amp�B�u�                                    Bx}~��  �          @c33?�{@H�ÿ��\���
B��f?�{@R�\�B�\�L(�B���                                    Bx}~��  �          @_\)?У�@Dz�>��
@�
=BxG�?У�@8��?�=qA��\Brz�                                    Bx}2  �          @w
=@'
=@5�>���@�B==q@'
=@*=q?�  AqG�B6=q                                    Bx}�  �          @i��@Q�@*=q?(�A  B@�H@Q�@�H?��\A�{B5��                                    Bx}!~  �          @C33@�
@ ��?^�RA�33B1=q@�
?�(�?�{A֣�B��                                    Bx}0$  �          @6ff@z�?�p�>B�\@|��Bff@z�?��?\)A<(�A���                                    Bx}>�  �          @N�R@�?�p�?\A�Q�Bff@�?��?�z�B{A�ff                                    Bx}Mp  �          @`��?8Q�@L(��.{�9��B��{?8Q�@G�?(��A9B�{                                    Bx}\  T          @c33>�=q@?\)=�Q�?��B��\>�=q@7�?\(�A�33B��                                    Bx}j�  T          @�  �)��@]p������
=B��f�)��@{�����s\)B�\                                    Bx}yb  �          @����*=q@e��z���Q�B�B��*=q@�녿����s�B�z�                                    Bx}�  �          @�z��\)@S33�&ff� �B�
=�\)@w
=��33��z�B��{                                    Bx}��  �          @�=q��@C�
�C�
���B�L���@p  �
=q��
=B���                                    Bx}�T  �          @����-p�@8���9����RC��-p�@c33��\����B��R                                    Bx}��  �          @�z��!G�@8���AG����C  �!G�@e��
=q�ѮB�(�                                    Bx}   �          @����!�@=p��@  �Q�C�=�!�@h���
=��  B���                                    Bx}�F  �          @����333@ff�Fff�!33C  �333@E�����\)CB�                                    Bx}��  �          @����@��?�p��Fff�"�HC�)�@��@.�R�{��=qC	Ǯ                                    Bx}�  �          @���0  @
=�L���+
=C��0  @8Q��!���
C��                                    Bx}�8  �          @�
=�1�?�ff�<(��&G�C�1�@!G��ff� �C	��                                    Bx}��  �          @����,(�@�R�-p��
=C	aH�,(�@G
=������33C��                                    Bx}��  �          @�  �&ff@   �C�
�,
=CW
�&ff@0  �=q�G�Cc�                                    Bx}�)*  �          @�\)�z�?���Vff�R�
CW
�z�?��<(��1ffC\)                                    Bx}�7�  �          @�p���\?���b�\�T��C@ ��\@p��C33�/=qC\                                    Bx}�Fv  �          @�ff�  ?�(��g��[  Ck��  @	���J=q�5C=q                                    Bx}�U  �          @���@  �L(��1�HC\�@A��p��
=B�W
                                    Bx}�c�  �          @�Q��!�?��J�H�3��C�)�!�@,���"�\�	�\C!H                                    Bx}�rh  �          @�\)� ��@$z��E�"��CO\� ��@S�
����=qB�G�                                    Bx}��  �          @�=q�#�
@5�:�H���C!H�#�
@aG���\��
=B��                                    Bx}���  �          @���1�@-p��8����C���1�@Y���33��CQ�                                    Bx}��Z  �          @�{�QG�@���=p���C��QG�@;��  ����C
#�                                    Bx}��   �          @�ff�HQ�@	���J=q�33C}q�HQ�@;������=qC�{                                    Bx}���  �          @�(��?\)@  �HQ���RC��?\)@AG������33C�q                                    Bx}��L  �          @�=q�   ?��H�G
=�6�C�
�   @   �!G��C��                                    Bx}���  �          @����,��?����j=q�HG�CǮ�,��@���G��#33C
T{                                    Bx}��  �          @�ff�(Q�?У��u��L{C:��(Q�@(Q��N�R�#��C{                                    Bx}��>  �          @���4z�?�z��z=q�L��Cu��4z�@(��W��(z�C&f                                    Bx}��  �          @��R�.{@z��XQ��-�C��.{@J�H�&ff��G�C��                                    Bx}��  �          @����0��?����xQ��JQ�C���0��@'��Q��#(�C�\                                    Bx}�"0  �          @�  �)��?����{��O=qC��)��@(Q��U��&�C33                                    Bx}�0�  �          @�
=�Q�@��vff�K��C���Q�@A��HQ��ffC :�                                    Bx}�?|  �          @�Q����@��n�R�@\)C�����@R�\�:�H�B�B�                                    Bx}�N"  �          @���G�@Q��c33�<��C���G�@Q��.�R�	��B�aH                                    Bx}�\�  �          @��H��@z��:�H�4�Cn��@3�
�{�z�B���                                    Bx}�kn  �          @�(���R?��
�>{�8CxR��R@#33�ff�  C:�                                    Bx}�z  �          @�=q��R?ٙ��9���7��CǮ��R@���33��CY�                                    Bx}���  �          @u��
=?�(��5��=�C
O\��
=@p��{�p�C :�                                    Bx}��`  �          @h�ÿ���@���z�� ��B�z����@5���=q�У�B���                                    Bx}��  �          @q녿޸R@(����
=B��f�޸R@?\)���R��  B�Q�                                    Bx}���  �          @`�׿���@p��\)�33B�=����@@  ��Q���G�B�u�                                    Bx}��R  �          @a녿��\@���  ��B��ÿ��\@<(����H��ffB���                                    Bx}���  �          @TzῨ��?���H�<B�  ����@������HB��)                                    Bx}���  �          @J=q��33?����  �6�C
���33@�\��p���C�                                    Bx}��D  �          @:�H��  ?�����z��)�C�q��  ?�Q쿴z���(�B�z�                                    Bx}���  �          @�H��  ?�
=��(��=qC�Ϳ�  ?Ǯ�����\)C�3                                    Bx}��  �          @����G�?�{�����!
=C녿�G�?�G���{����B�p�                                    Bx}�6  �          @E��޸R?�{�˅� =qC�޸R@\)�z�H����B�p�                                    Bx}�)�  �          @4z��z�@녿}p����CG���z�@�R��Q���G�B�L�                                    Bx}�8�  �          ?�(����\?��R�z����C�
���\?��;B�\�љ�B�(�                                    Bx}�G(  �          @W
=��{@	�������C�H��{@"�\�p����ffB��=                                    Bx}�U�  �          @dz��=q?���p��/��C�\��=q@(���=q��B��                                    Bx}�dt  �          @���%?��L���4�RC���%@,��� ���ffC�=                                    Bx}�s  �          @�  �0��?�=q�aG��?p�C(��0��@#�
�8���(�C	8R                                    Bx}���  �          @��� ��?�
=�`���J{CJ=� ��@=q�:�H� =qC+�                                    Bx}��f  �          @�p���R?�Q��p  �U33C�
��R@.�R�Dz��$�\C=q                                    Bx}��  �          @��H�
=?��xQ��U�C���
=@0���Mp��%��C��                                    Bx}���  �          @�
=�Q�?��\��Q��i33C8R�Q�@=q�\���<
=Cu�                                    Bx}��X  �          @�G���?��
���\�i��CǮ��@���e��B
=C�                                    Bx}���  �          @�
=��?��
��Q��W33C����@:=q�R�\�%�RC �R                                    Bx}�٤  �          @�{�/\)?�{��ff�W�RC���/\)@#�
�e�/(�C�R                                    Bx}��J  �          @��
�0  ?�G���(��V�CxR�0  @(��c33�/�C
h�                                    Bx}���  �          @�z��!�?�G���33�gCT{�!�@G��u�A��C
�                                    Bx}��  �          @���!�?�\)����f�C��!�@���s�
�>�\C�)                                    Bx}�<  �          @�z��  ?s33��Q��vffC���  @���Q��M�\C�=                                    Bx}�"�  �          @�33��R?^�R��  �w�
C���R@�������P(�Cff                                    Bx}�1�  �          @�  ��?Tz���p��xz�C
��@	���|���Q  C��                                    Bx}�@.  �          @��
��p�?z������
C#����p�?�33��Q��^��C&f                                    Bx}�N�  �          @�(����>�  ����C,h����?�\)��{�m=qCs3                                    Bx}�]z  �          @�zῺ�H�B�\���C;Q쿺�H?�G������C33                                    Bx}�l   �          @�p�����>k���  ��C,������?�{���R�o{C
=                                    Bx}�z�  �          @����      ��
=  C4���  ?�����  �
C	O\                                    Bx}��l  �          @�33����Ǯ��{8RCG�H���?�����\�)C5�                                    Bx}��  �          @�33�����
��  C4�q��?�Q���ff�fC)                                    Bx}���  �          @�����?B�\���(�C\���@���ff�oz�B�33                                    Bx}��^  �          @�(���?J=q��p�Q�C�῕@p���{�l��B��H                                    Bx}��  �          @�Q쿵?z���Q�p�C�H��?��H��33�m��B���                                    Bx}�Ҫ  �          @��R�˅=��
��{\)C1��˅?�ff��p��}=qC�{                                    Bx}��P  �          @����������aHCG���?u���\ǮC!H                                    Bx}���  �          @��
��=q�#�
����qC48R��=q?�z������v\)CY�                                    Bx}���  �          @�����>�����|��C)�{���?��
��=q�\��C                                    Bx}�B  �          @��H��(��L����Q�  C:����(�?��R���H���C@                                     Bx}��  �          @�33�ff>\�tz��hQ�C*ٚ�ff?����`  �L\)C8R                                    Bx}�*�  �          @���1G�?333�o\)�RG�C%�=�1G�?����Tz��3p�C:�                                    Bx}�94  �          @����33>k�����C-�\�33?˅��Q��dz�C
                                    Bx}�G�  �          @����?E�����y��C�=��@�qG��P
=C��                                    Bx}�V�  �          @�{��?Q��}p��h=qC �R��@�\�_\)�A�C
�=                                    Bx}�e&  �          @�Q��.{?!G��vff�XffC'��.{?�=q�\���9��C
=                                    Bx}�s�  �          @�p��,��?�=q�|(��V33C&f�,��@33�X���.�C��                                    Bx}��r  �          @����?\)?��
�k��>�\C޸�?\)@(Q��?\)��\C
��                                    Bx}��  �          @�G��N{?�=q�c�
�633C� �N{@=q�;��Q�C.                                    Bx}���  �          @����J=q?5�s�
�FffC'W
�J=q?�33�W��)��C�                                    Bx}��d  �          @�p��U�?h���u�@�C$Ǯ�U�@ff�U�!
=C��                                    Bx}��
  �          @��\�dz�>��H�c�
�2�
C,��dz�?У��Mp��=qC}q                                    Bx}�˰  �          @�  �E?s33�s33�F=qC"���E@Q��Q��$p�Cff                                    Bx}��V  �          @���Z=q?����]p��0\)C"�H�Z=q@���:�H��C�f                                    Bx}���  �          @�=q�Q�?G��q��A(�C&���Q�?�(��Tz��#�C�3                                    Bx}���  �          @��\�Vff?fff�mp��;�C$��Vff@z��L����CG�                                    Bx}�H  �          @����Z�H?xQ��e��4C$#��Z�H@ff�C�
�\)C��                                    Bx}��  �          @�{�"�\>���{�iffC)���"�\?����s33�JQ�CaH                                    Bx}�#�  �          @���@  ?(��mp��I�HC(�=�@  ?�ff�S33�-Q�C�                                    Bx}�2:  �          @����:=q>L���qG��Q(�C0\�:=q?�Q��_\)�<ffC�3                                    Bx}�@�  �          @�
=��{=�Q��tz�#�C0Ǯ��{?����e��m��C�                                    Bx}�O�  �          @|�Ϳh�þ�=q�s�
\CD�
�h��?�G��k�z�C�f                                    Bx}�^,  �          @Mp����;�z��B�\¡�=CWJ=����?@  �>{�B��f                                    Bx}�l�  T          @��\��
=?����z�k�C�쿷
=@(���\���C��B�                                    Bx}�{x  �          @���(��aG����\.Ck  �(�?�R���
aHC�                                    Bx}��  �          @�(��8Q�.{��Q�L�C_�=�8Q�?^�R��\)�RCc�                                    Bx}���  �          @����O\)�\)����HCV�\�O\)?s33���H�fC33                                    Bx}��j  �          @�녿�?
=q��33�CQ쿕?�(��z�H�n�B�G�                                    Bx}��  �          @�녿�녾�����z�
=CB�ῑ�?������u�C}q                                    Bx}�Ķ  �          @��ÿ\(��^�R��33�\Ca� �\(�?&ff��z�p�Cٚ                                    Bx}��\  T          @�G��k��J=q���
\C\� �k�?=p���(��C!H                                    Bx}��  �          @�G���=q���
����� C4녿�=q?�p���z�aHB��                                    Bx}��  �          @��ÿ�\)������p�� C7�f��\)?����}p��x��C�q                                    Bx}��N  �          @�녿��R�Tz������RCU���R?0�����\�C�                                    Bx}��  �          @��ÿ��
�����Cd�ῃ�
>�33��z�=qC!                                    Bx}��  �          @��ÿ���\(����B�C[ٚ���?+���33aHCG�                                    Bx}�+@  �          @��R�333�W
=��G� �{CD�)�333?��\��33�fB�                                    Bx}�9�  �          @���Tz������\)CT.�Tz�?}p���ff��C�                                    Bx}�H�  �          @�33������
��G�aHC?���?���z=qW
C�q                                    Bx}�W2  �          @��\����5���
�CU�
���?B�\����3C�                                    Bx}�e�  �          @�z�8Q����  \CU���8Q�?��\�����B�(�                                    Bx}�t~  �          @��R�u?������\C&f�u@ ���u�o�\B�.                                    Bx}��$  �          @����?z�H���\=qC���@�^�R�Q��B�ff                                    Bx}���  �          @����G�?���|���C#׿�G�@+��N{�=p�B�\)                                    Bx}��p  �          @�ff�\(���33���\��CJ^��\(�?�
=���=B�{                                    Bx}��  �          @�  ���׽�G�����p�C9�׿���?�33��=qQ�C ��                                    Bx}���  �          @�{�J=q�k����\�CD�=�J=q?�ff��z��3B�                                     Bx}��b  �          @���^�R>\)����C+  �^�R?���\)\B�                                    Bx}��  �          @�33��\)>\)��p��RC,���\)?����w
=�|�B���                                    Bx}��  �          @�zῢ�\�.{��ff\C;�쿢�\?����\)C��                                    Bx}��T  �          @��\��  ������\u�CM�ῠ  ?aG������C�                                    Bx}��  �          @��Ϳ���������
Q�C`W
���>���\)�RC33                                    Bx}��  �          @����
��  ���
  CY�f���
?���ff�=C��                                    Bx}�$F  �          @��Ϳ�G����
���\u�C[h���G�>���B�C�                                    Bx}�2�  �          @�33��(���=q��Q��)C]^���(�>�
=��(�G�C �f                                    Bx}�A�  �          @�����33�h����G�#�CZ����33?
=��33�C�)                                    Bx}�P8  �          @�=q�J=q��p���  ffCq�q�J=q=#�
����  C0��                                    Bx}�^�  �          @��Ϳ��Ǯ���C{aH���#�
��33¦33C4Ǯ                                    Bx}�m�  �          @�z�0�׿�ff��G�  Cq�H�0��>u��� aHC!
=                                    Bx}�|*  �          @�p������.�R�U�I��C�˅���Ϳ������H#�C}Q�                                    Bx}���  �          @���u��\�u�yffCqh��u��=q��
=\)CCs3                                    Bx}��v  �          @��?E��&ff�J�H�F
=C�>�?E����
�y��{C��H                                    Bx}��  �          @�=q?���8Q��HQ��/{C�
=?���Ǯ�}p��tG�C�Q�                                    Bx}���  �          @��H?��
��\)�q��^�\C���?��
�\��ff#�C��                                    Bx}��h  �          @�Q�?����{��  {C�=q?��>8Q���
=��@��                                    Bx}��  �          @�G�=�G���=q�����\C�Z�=�G�>k���  ­{B���                                    Bx}��  �          @��;�  �
=�tz��s��C��R��  ����=q£z�CvW
                                    Bx}��Z  �          @��׾W
=��Q��u­G�CKs3�W
=?����g
=u�B��H                                    Bx}�    �          @��?�z�@h�þL���9��B��{?�z�@]p�?�33A�Q�B��{                                    Bx}��  �          @��?��
@5�p���B��=?��
@\�Ϳ��\�x(�B�aH                                    Bx}�L  �          @�{?�=q@��HQ��C�Bs�?�=q@W
=��\��B���                                    Bx}�+�  �          @��׽L��>aG���  ®B�Bή�L��?�{���L�B�33                                    Bx}�:�  �          @�33=�Q�>�z���G�¬W
B��=�Q�?�Q����u�B���                                    Bx}�I>  �          @��׽��
=�Q���Q�±CQ콣�
?�  ���k�B�                                      Bx}�W�  �          @�Q���u��  ­.Cr�=��?�Q�������B��q                                    Bx}�f�  
�          @�z���8Q����� ��C�׾�?k����k�BøR                                    Bx}�u0  �          @s�
�\)�z�H�k�ǮC���\)>�ff�q�¥�)Bճ3                                    Bx}���  �          @��W
=�
=q��z�¥  Cx� �W
=?��������fB�=q                                    Bx}��|  �          @���>����H�l(�ǮC�%>�<��
�~{¦�@ff                                    Bx}��"  �          @�{>.{��������Q�C���>.{>\��¨�B�z�                                    Bx}���  �          @\)>����
=�n�R{C�.>��>����z=q¨z�BCG�                                    Bx}��n  �          @y��@ff�p�����\)C�� @ff��p��7��F�HC��{                                    Bx}��  �          @��
?!G����
�j�H�{G�C��{?!G���=q��� �C�5�                                    Bx}�ۺ  T          @�\)?:�H��33�l(��t\)C�z�?:�H�\��(�� C�=q                                    Bx}��`  �          @�G�?�  �z��hQ��f��C�޸?�  �\)����(�C�u�                                    Bx}��  �          @���?xQ��
=�mp��oQ�C�S3?xQ�Ǯ��p��)C�
=                                    Bx}��  �          @��
?L���
�H�l(��g�
C�3?L�Ϳ�R��\)B�C�)                                    Bx}�R  �          @��
�u�"�\�a��Y
=C�\)�u���\��\)�HC�g�                                    Bx}�$�  �          @�p��^�R���g��^{Cy�\�^�R�J=q��\)
=C^0�                                    Bx}�3�  �          @��H?
=��33�z=q�qC���?
=�L����  ¤=qC��H                                    Bx}�BD  �          @�=q>�p���\�xQ�  C���>�p��.{����©8RC���                                    Bx}�P�  �          @���?&ff�Tz����
Q�C�(�?&ff?L�����
�
BKz�                                    Bx}�_�  T          @�=q?���ff��  ¡��C���?�?�����33�\B�Q�                                    Bx}�n6  �          @�(�?!G��   �`  �lz�C��?!G���\��  C���                                    Bx}�|�  T          @�
=?�R�:=q�S33�?�
C��?�R��
=�����C��=                                    Bx}���  T          @�p�?�\�6ff�Tz��C�C�?�\��{����z�C�:�                                    Bx}��(  �          @���>�  �K��4z��&
=C�E>�  �����u��C��                                     Bx}���  �          @�
=>.{�\���2�\���C�k�>.{�
=�y���u��C�O\                                    Bx}��t  �          @��\?��0  �fff�PQ�C�y�?���z���(���C�\)                                    Bx}��  �          @������H�qG��e  C��
���G������C���                                    Bx}���  �          @�z᾽p���H�g
=�_��C�����p��W
=��Q���CvY�                                    Bx}��f  �          @�p���
=���w��u�C�@ ��
=��
=���
¤u�C`�)                                    Bx}��  �          @��
�u��ff�}p��C�9��u����33¬�)CPG�                                    Bx}� �  �          @����\��o\)�q��C��q�\����Q�£�RCg��                                    Bx}�X  �          @�  @
=q�p�׿�=q�`��C��@
=q�B�\�p����C���                                    Bx}��  T          @��?�(��7��?\)�+G�C���?�(���p��w��v��C�c�                                    Bx}�,�  T          @�
=?��Ϳ�\�Y���[�C�?��;��R�tz���C���                                    Bx}�;J  �          @�ff?�{���S�
�P�HC���?�{�5�y��  C�+�                                    Bx}�I�  �          @�ff?�녿����Z=q�[��C��{?�녾��y���C��f                                    Bx}�X�  T          @��?�=q��  �X���dffC��q?�=q��\)�s�
�qC��                                    Bx}�g<  �          @��\?�(����>{�;�C��=?�(����\�j=q���C���                                    Bx}�u�  �          @��
?��
��\�J�H�I�\C���?��
�\(��s�
�)C�                                      Bx}���  �          @��
?�{��G��\���ep�C�c�?�{���p���3C�aH                                    Bx}��.  �          @�(�?��Ϳ��g
=�t��C��?���>�{�q�\AA�                                    Bx}���  �          @�  ?�
=��z��x���
C��q?�
=>������A�{                                    Bx}��z  �          @�Q�?��׿����q��z�C���?���>k�������A=q                                    Bx}��   
�          @���?˅����qG��z�C�:�?˅>�G��z=q�Ay�                                    Bx}���  T          @�ff?�녿�Q��fff�|(�C�&f?��=�G��xQ�8R@��H                                    Bx}��l  T          @�G�?����R�dz��d��C�4{?������=q.C�Z�                                    Bx}��  �          @���?�����dz��dQ�C��?����{��G�u�C���                                    Bx}���  
�          @��\?�
=��=q�j=q�g  C�c�?�
=<#�
�~�R8R>.{                                    Bx}�^  �          @�Q�?�G���(��\(��X��C���?�G���G��|(�u�C��f                                    Bx}�  �          @��?xQ쿊=q�{�aHC�
=?xQ�?z�������A�\)                                    Bx}�%�  �          @��R=L�Ϳ�ff����ǮC�s3=L��?&ff��(�¢B���                                    Bx}�4P  �          @�ff?@  �����s33��C��3?@  >L�����ffAiG�                                    Bx}�B�  �          @�Q�@8Q���R�z���33C�!H@8Q쿚�H�2�\�'(�C��
                                    Bx}�Q�  �          @��@   �����=p��/  C��
@   ���\(��V{C���                                    Bx}�`B  �          @���@��\�C�
�9{C�xR@�Ǯ�`���`  C�O\                                    Bx}�n�  �          @�33@Mp��	����(��׮C�  @Mp����*=q���C���                                    Bx}�}�  T          @�z�@I����(����\C�5�@I�����
�5�"��C��                                    Bx}��4  �          @�p�@\�Ϳ��ٙ���p�C��)@\�;Ǯ�����
C��f                                    Bx}���  �          @��
@���>�Q����z�@��\@���?
=q�������@�ff                                    Bx}���  �          @�{@�(��녾B�\� ��C�O\@�(���G��������C�!H                                    Bx}��&  T          @���@�ff�h�ý��
���
C�>�@�ff�L�;�G����\C��                                    Bx}���  T          @���@��ÿ��\>��H@�
=C��R@��ÿ�{��\)�h��C�.                                    Bx}��r  T          @���@\)���
��{�dz�C���@\)���޸R���C���                                    Bx}��  �          @���@�=q��=q��(���p�C��\@�=q��z�� ���љ�C��R                                    Bx}��  
�          @�=q@�Q�\�\��  C���@�Q�E�� ����p�C���                                    Bx}�d  �          @�G�@h���	�����H��{C���@h�ÿ�(���p���ffC��                                    Bx}�
  �          @�Q�@h����Ϳ�\)��z�C�j=@h�ÿ����	����C�                                      Bx}��  �          @��\@tz��G���G��M�C���@tz��z������C�C�                                    Bx}�-V  �          @���@[���=q�.{�
=C�n@[���Q��A��%�C�4{                                    Bx}�;�  �          @�  @hQ������{C�C�@hQ�B�\�:=q�ffC��                                    Bx}�J�  �          @��@n{���H�   � G�C�>�@n{���R�8Q���HC���                                    Bx}�YH  �          @�p�@QG�����H���(�HC�ff@QG�>�  �U��5��@�                                    Bx}�g�  �          @�
=@k���  �0���C�]q@k�>aG��;��
=@^{                                    Bx}�v�  �          @��\@fff�u�)���z�C��f@fff>W
=�3�
��\@Z=q                                    Bx}��:  �          @��H@c33�E��2�\�33C��f@c33>�ff�7
=��@��                                    Bx}���  �          @���@S33�@  �A��'ffC��{@S33?��Dz��*�A(�                                    Bx}���  T          @��
@n�R�.{�-p���
C���@n�R?xQ��!��Q�AiG�                                    Bx}��,  �          @��@p  ����(�����C���@p  ?8Q��$z���
A,��                                    Bx}���  �          @�z�@\�;�ff�Dz��%�
C�Ff@\��?\(��>�R�   A`��                                    Bx}��x  �          @��
@8�ÿ����Z=q�>��C�^�@8��>Ǯ�dz��J�@�\)                                    Bx}��  �          @��
@8�ÿfff�^�R�D
=C�W
@8��?!G��b�\�H��AD(�                                    Bx}���  �          @�p�@S33��  �N�R�1=qC��\@S33?�\)�B�\�$�\A�\)                                    Bx}��j  �          @��
@?\)�\)�\���Bp�C���@?\)?n{�W��<�A��                                    Bx}�	  �          @��H@#�
�xQ��j�H�U=qC��f@#�
?&ff�n�R�[{Ae�                                    Bx}��  �          @��\@p��z�H�hQ��Wz�C�%@p�?!G��mp��^Q�Ad(�                                    Bx}�&\  �          @��
@'��Y���g
=�R�RC�  @'�?=p��hQ��T��A{�                                    Bx}�5  �          @��\@0�׾��fff�P�C�8R@0��?���^{�E��A�                                      Bx}�C�  �          @�=q@5��=q�c33�L�C�O\@5?�  �U�<\)A��                                    Bx}�RN  �          @���@'�>��j�H�Z
=@<��@'�?�z��Q��:\)B�
                                    Bx}�`�  �          @��?�
=>\)�y���~ff@�G�?�
=?�G��^�R�TffB)�\                                    Bx}�o�  �          @�
=?^�R�#�
��33  C��{?^�R?�Q��n�R�|  B{�                                    Bx}�~@  �          @�{?����(��|��z�C���?���?�\)�u�
B+                                    Bx}���  �          @�p�?��
����i���{�\C�s3?��
?���b�\�n�A��
                                    Bx}���  �          @�
=?���(��fff�up�C��R?�?����\���dffA�=q                                    Bx}��2  �          @��?�z΅{�p���z(�C�
=?�z�>�Q��~�R�fAe                                    Bx}���  �          @�p�?�ff�@  ����C�{?�ff?����    Bz�                                    Bx}��~  �          @�{?�����\)L�C��?�?�������BA                                    Bx}��$  �          @���?Ǯ�\(����\C��R?Ǯ?����33��B�                                    Bx}���  �          @�?���{��Q��HC�1�?�?0������A�                                    Bx}��p  �          @�z�?�zῐ������z�C��H?�z�?fff���H�HB=q                                    Bx}�  �          @�p�>��H�+����\aHC��>��H?����aHB��f                                    Bx}��  �          @�33?\)�xQ����R33C�f?\)?�����p��)B}�                                    Bx}�b  �          @��?�G����\�����\C�� ?�G�?h����=qp�B��                                    Bx}�.  �          @��R?�Q�aG����RQ�C���?�Q�?�{�����B+�                                    Bx}�<�  �          @�p�?8Q쾙�����H �3C��R?8Q�?�Q������B�Ǯ                                    Bx}�KT  �          @��H?Tz�   ��\)�3C���?Tz�?�(���Q��)BrQ�                                    Bx}�Y�  �          @��?5�k���{��C��{?5?�������=Baff                                    Bx}�h�  �          @��
?�\)���g
=�l\)C���?�\)=#�
�~{  ?��R                                    Bx}�wF  T          @�33?��
�&ff�G��7�C�<)?��
��G��{��C�@                                     Bx}���  �          @��H?�  �����mp��m
=C�/\?�  �L����(�p�C�Ф                                    Bx}���  �          @��\?@  ��\)���\k�C�Ф?@  ?=p�����B2�
                                    Bx}��8  �          @�=q>���z���\)¢�HC�.>��?�ff��=q��B��                                    Bx}���  �          @�\)����=������©p�C&�����?��R�|(��{{Bʅ                                    Bx}���  �          @�
=�Ǯ>\)��©L�C ��Ǯ@��|(��y
=B�Ǯ                                    Bx}��*  �          @��?.{��(����HǮC��?.{?����w��RB{                                    Bx}���  �          @���@]p���R����z�C��3@]p���(��'��=qC�H�                                    Bx}��v  �          @���@@  ��{�(Q����C�q@@  �
=q�K��8�
C��                                    Bx}��  �          @�33@`�׿�����\��  C�R@`�׿.{�7
=��HC���                                    Bx}�	�  �          @��H@Q녿�G��2�\��RC���@Q녾���J�H�/��C��                                    Bx}�h  �          @��\@;����R�L(��0��C��=@;�=#�
�aG��H�H?Tz�                                    Bx}�'  �          @��\@,(����9�����C��H@,(��J=q�fff�P\)C��=                                    Bx}�5�  �          @���@'������HQ��2{C��q@'���\)�g
=�W�C���                                    Bx}�DZ  �          @��
@U����Dz��(�
C�%@U�?Q��@���%  A]�                                    Bx}�S   T          @�33@7�@(��(���p�B�@7�@G�����(�B=G�                                    Bx}�a�  T          @���@0��?�z��X���B(�A�
=@0��@!G��"�\��B)�
                                    Bx}�pL  �          @�G�@<(�?(��W
=�@A;�@<(�@ ���0����HB	�\                                    Bx}�~�  T          @�G�@\(��L���:�H�!p�C���@\(�?�p��)�����A��                                    Bx}���  �          @���@c33�(��*=q���C�'�@c33?�R�*=q���A��                                    Bx}��>  �          @�\)?�����dz��W�C��q?��\)������C��                                    Bx}���  �          @�{@{�(��Fff�3G�C���@{�(��o\)�ip�C�T{                                    Bx}���  �          @�@���Fff�2�RC�(�@���l���d33C�Ǯ                                    Bx}��0  �          @�@5��>�R��p���z�C�� @5����(����\C�*=                                    Bx}���  �          @���@/\)�!������C���@/\)�����@���2�\C��                                    Bx}��|  �          @��@%��?\)��\)����C�Z�@%���p��1G��"  C�>�                                    Bx}��"  �          @��@'�������ffC���@'������N{�AC�]q                                    Bx}��  �          @���?}p��+��N�R�A��C�(�?}p��}p����\�C���                                    Bx}�n  T          @�\)?����@���'����C�}q?�����ff�k��_�\C�˅                                    Bx}�   T          @��?��
�X���%��C�T{?��
��z��tz��k��C��                                    Bx}�.�  �          @��R?���c33���  C��H?������l(��c
=C���                                    Bx}�=`  �          @��
@?\)�����(��	\)C�n@?\)�.{�C33�3p�C���                                    Bx}�L  �          @��@P�׿�
=����C��@P�׼��'��(�C���                                    Bx}�Z�  �          @��\@
=�7
=�p���G�C�4{@
=�˅�P  �K��C���                                    Bx}�iR  �          @�p�@3�
�<�Ϳ�p����C��3@3�
����6ff� �C�q                                    Bx}�w�  �          @�ff?޸R�Y���   ��33C���?޸R�
�H�S33�GC�\)                                    Bx}���  �          @�=q?���~{���H��Q�C��q?���333�R�\�;�C�Z�                                    Bx}��D  �          @�p�?�33�z�H����(�C�h�?�33�*=q�]p��?G�C��H                                    Bx}���  �          @��?���\)�   ��Q�C���?���+��c�
�HffC�ٚ                                    Bx}���  T          @�p�?���'
=�?\)�+�C��?�׿�G��u�s��C��=                                    Bx}��6  �          @�z�?��&ff�AG��.�RC�t{?��z�H�w
=�w��C���                                    Bx}���  �          @�(�?����R�A��0�C��R?�׿aG��tz��u�C���                                    Bx}�ނ  �          @�\)?����
=�O\)�:�C��=?����+��|���yC���                                    Bx}��(  �          @�\)?��C33�)����C�)?���ff�n�R�a�
C��=                                    Bx}���  �          @�  ?�\�G
=�.�R�{C���?�\�����u��i  C�4{                                    Bx}�
t  �          @���?�33�C33�0  ��C��)?�33��  �tz��fffC�ٚ                                    Bx}�  �          @�Q�?��H�>{�1���
C��)?��H���s�
�fG�C�{                                    Bx}�'�  �          @���?�{�G
=�6ff��C��?�{��G��|(��r�C�h�                                    Bx}�6f  �          @�?Ǯ�G��.{��C�O\?Ǯ�����u��o�RC�h�                                    Bx}�E  �          @���?�{�Tz��0  ���C�:�?�{�޸R�|(��y�RC�B�                                    Bx}�S�  �          @�p����
�b�\�Y���/�\C�c׽��
������
�C���                                    Bx}�bX  �          @��R�#�
�e��Y���.
=C����#�
��
=��z��C�7
                                    Bx}�p�  �          @���=�Q��qG��Q��$�C���=�Q��33��(�B�C�T{                                    Bx}��  �          @�>��
�L(��n�R�E\)C���>��
��
=����G�C��=                                    Bx}��J  �          @�Q�>��
�J�H�vff�I�C��=>��
��{���W
C�'�                                    Bx}���  �          @��
?���R�\�u��Cz�C�4{?����(�������C�.                                    Bx}���  �          @��R?(��b�\�o\)�8�C�޸?(���p���{C�&f                                    Bx}��<  �          @�=q=�\)�ff��33�C���=�\)>B�\����¯Q�B�\                                    Bx}���  �          @�녽�G�����ff8RC�>���G�>������¨�B�Ǯ                                    Bx}�׈  �          @�
=���=#�
��ff�{C2Y����@�������k�B�.                                    Bx}��.  �          @�{���=�G���Q�� C/:ῧ�@ff��p��j�\B�W
                                    Bx}���  �          @�������
=\C8�῵@������q�B�(�                                    Bx}�z  �          @�
=�˅>#�
���R8RC.^��˅@Q�����b��B��                                    Bx}�   �          @�z῝p�>�p����R��C#+���p�@$z���Q��`�HB�#�                                    Bx}� �  �          @�Q�
=q�����ff¦.CM(��
=q@ �������3B�G�                                    Bx}�/l  �          @�\)��R��\)��p�¥� C:�
��R@
=q���|ffB�(�                                    Bx}�>  �          @��H�s33?+����(�C�\�s33@333��G��W(�B�W
                                    Bx}�L�  �          @�Q�����
��\)�}
=C~�ÿ��=��
���R¦u�C,
=                                    Bx}�[^  �          @�=q��33���
���\33C4�
��33@
�H��=q�q\)B��)                                    Bx}�j  �          @�ff����?O\)����HC�)����@<����Q��K�
B�=q                                    Bx}�x�  �          @�33�?�33�����h{C	���@o\)�J�H��HB�B�                                    Bx}��P  �          @��H��@   ����j�HC�\��@vff�J=q�33B���                                    Bx}���  �          @�녿��@(���G��k�\B�Ǯ���@�Q��C33���B��{                                    Bx}���  �          @��
���@\)����mp�B�q���@~{�7��	�B��)                                    Bx}��B  �          @�  ��z�@!����H�_�B�.��z�@���   ��ffB�u�                                    Bx}���  �          @���   ?�����H#�B��   @H���QG��7=qB�33                                    Bx}�Ў  �          @��\=�Q�?8Q��|���B�aH=�Q�@(��L(��R\)B���                                    Bx}��4  �          @�����@fff�Y���2�RB��)��@e?aG�A:ffB���                                    Bx}���  �          @���<��@qG�<��
>8Q�C 
�<��@Vff?�p�A�G�Ch�                                    Bx}���  
�          @�ff�5@�  ?(�@���B����5@R�\@
=A�=qC�=                                    Bx}�&  �          @��
� ��@Z=q?���A���B�� ��@=q@1�B�C=q                                    Bx}��  T          @��ÿE�?5��Q��)C	k��E�@p��P  �N�BָR                                    Bx}�(r  �          @\��>L��>�\)�W�¨B�BWz�>L��?��
�7��g�
B��                                    Bx}�7  �          @��?(��\�����C��R?(�?
=q����\B'{                                    Bx}�E�  �          @��׿0����������C�4{�0���:=q�U�@ffC�H�                                    Bx}�Td  �          @��
��=q�xQ��G����HC~����=q��H�r�\�\33Cv                                    Bx}�c
  �          @�(���\)�U��0����
Ct  ��\)�ٙ��~{�m��CbQ�                                    Bx}�q�  �          @��
��  �(Q��`  �H�Ct�=��  �@  ��=q��CR��                                    Bx}��V  �          @��\��Q��j�H��
=��p�Cr
��Q��Q��X���?Q�Cf�=                                    Bx}���  �          @���:�H�J=q�g��@�RC����:�H��33��p�u�Cm��                                    Bx}���  �          @���s33�33���\�jffCwz�s33�8Q����{C>��                                    Bx}��H  �          @��
=�(�����   CbͿ
=?�z�����B�{                                    Bx}���  �          @�(��5?����H�fB�\�5@W
=�[��3�
B�Ǯ                                    Bx}�ɔ  T          @�����33?8Q���ff ��B�=��33@2�\�vff�W33B�8R                                    Bx}��:  �          @�G�?�ff?�ff��z���B_ff?�ff@X���L(��'�\B�u�                                    Bx}���  �          @���?�33>\��(�Q�A4  ?�33@�n{�K�
BK��                                    Bx}���  �          @�G�?G�@\)�\)�c(�B�33?G�@�G������  B�#�                                    Bx}�,  �          @��?}p�@(��s�
�]p�B��?}p�@z�H�G���z�B��                                    Bx}��  �          @�������h���z�H�{C{�3����?}p��y��8RBը�                                    Bx}�!x  �          @�Q�
=q�?\)�h���H�\C��f�
=q�}p���33��Cq\)                                    Bx}�0  T          @�������  ���
ǮC{
=��?\)��33 8RC�                                    Bx}�>�  �          @��H���
?0����Q�(�C�q���
@,(��l(��H33B�aH                                    Bx}�Mj  �          @�G����=L�����\
=C1�R���@����G��l�B��                                    Bx}�\  �          @�  >��������z�=qC��>��?!G����
£8RB��{                                    Bx}�j�  T          @�  >L�Ϳ�=q���ǮC��>L��>�Q���p�©��Bu=q                                    Bx}�y\  �          @�p�>�p�������(�B�C�3>�p�?:�H���B|�                                    Bx}��  T          @�\)?�=q�3�
�aG��B\)C��\?�=q�^�R��p�ffC�ff                                    Bx}���  �          @���?�
=�]p��@���
=C�5�?�
=��Q���Q��zQ�C�)                                    Bx}��N  �          @��
?�33�Fff�\���5�C�.?�33��z�����)C�1�                                    Bx}���  �          @��H?����Dz��P  �0
=C�!H?��׿�p�����z�C�:�                                    Bx}�  �          @��������G��s33�3�C�` �����s�
�C33���C�5�                                    Bx}��@  �          @��ý�Q���
=�����`  C�y���Q������#�
���C�c�                                    Bx}���  �          @��R?�
=��33����\)C��?�
=�P  �J=q�&C���                                    Bx}��  �          @�ff@�
���\�����C��@�
�>�R�G
=�{C��f                                    Bx}��2  �          @�?��R��\)��p����\C���?��R�Dz��QG��'
=C�p�                                    Bx}��  �          @�?�  ���H���\�AC��R?�  �fff�A���RC���                                    Bx}�~  T          @��R>�(���z��\���C�� >�(���33�-p���C���                                    Bx}�)$  �          @���?J=q��{�����C���?J=q��=q�=q��Q�C�0�                                    Bx}�7�  T          @�Q�?�Q������
�k�C��3?�Q���  ������HC�xR                                    Bx}�Fp  �          @�  ?�\�p�׿�Q���G�C��H?�\�*=q�@���-(�C�Ф                                    Bx}�U  �          @�p�@*=q�L(��G����HC��@*=q����_\)�=�RC�                                    Bx}�c�  �          @��\?}p���\��
=#�C�k�?}p�?�\)��ffp�Bjz�                                    Bx}�rb  �          @���?������������C���?��?����Q��v�
B\��                                    Bx}��  �          @�z�?\(�?z�H����3BB�\?\(�@AG��k��E��B�(�                                    Bx}���  �          @��?u@   ���\�bB�#�?u@�33�p���{B��q                                    Bx}��T  �          @�z�?\)?�  ��\)aHB�33?\)@g��J=q�#�B�G�                                    Bx}���  �          @��\    ?�G���\)B��    @h���J=q�#��B�
=                                    Bx}���  �          @���?��?0���z�H�=B?��@��I���GffB���                                    Bx}��F  T          @�=q@�׿�z��z=q�^��C��@��?\)���
�r{A^�H                                    Bx}���  �          @��\@��+��8���!�C��=@�����s33�j(�C��                                     Bx}��  �          @��?�
=����У���C��H?�
=�N�R�a��,�C�s3                                    Bx}��8  �          @��H@=q���\����A��C���@=q�U�<���(�C��f                                    Bx}��  �          @��@,(����
���H�]�C��\@,(��E�>�R�\)C���                                    Bx}��  ?          @���@&ff���R�����t��C�ٚ@&ff�Fff�I����C���                                    Bx}�"*  �          @�33@G���ff?#�
@�  C�/\@G�������R����C���                                    Bx}�0�  �          @��?�Q���=#�
>�C�G�?�Q���p��Q����HC��H                                    Bx}�?v  �          @��
@\)��Q�#�
���C�4{@\)�l���*=q���RC���                                    Bx}�N  T          @��H?ٙ����>���@X��C��q?ٙ����H��
=��(�C���                                    Bx}�\�  �          @��
@Q���ff�#�
�   C��@Q��z�H�ff����C���                                    Bx}�kh  �          @��\?�\)��z��\��=qC���?�\)�w��&ff��\)C��                                    Bx}�z  �          @��
?�����\��{���RC�k�?���0  �b�\�C�RC�j=                                    Bx}���  T          @�
=�s33�33�c�
�fQ�CuG��s33�.{�����C>&f                                    Bx}��Z  �          @�\)�9���&ff�g
=�JG�C@�f�9��?����`  �A��C:�                                    Bx}��   T          @�����p���z��z=q�hffCWn��p�?\)����}��C$&f                                    Bx}���  �          @�33��p��&ff�e�D  ClY���p��#�
��z�k�CH8R                                    Bx}��L  �          @��ÿ�33�dz��,����Cx�׿�33��z������n��Ci�{                                    Bx}���  �          @�녿�\�X���2�\�p�Cr}q��\���H��G��j��C`{                                    Bx}���  �          @��׾�  �C�
�(��C��=��  �ٙ��W
=�{�C��                                    Bx}��>  �          @��H?�{�w
=�
=q�ՙ�C�� ?�{��H�l���I�C���                                    Bx}���  �          @�?����h���<���33C���?��ÿ�{����xffC���                                    Bx}��  �          @��R@I���h�ÿ0���ffC�t{@I���:=q�33��C��                                    Bx}�0  �          @�\)����q��@  �G�C}�׿�����H����|G�Cp�
                                    Bx}�)�  �          @��׾B�\��  �)�����
C���B�\�!���=q�n\)C��\                                    Bx}�8|  �          @��=#�
��=q�	�����C�@ =#�
�C33�����S(�C�`                                     Bx}�G"  �          @��H�����������H����C�aH�����e��aG��2{C�+�                                    Bx}�U�  �          @�zἣ�
��G��Q���C�� ���
���H�C�
�z�C��R                                    Bx}�dn  T          @��>������׽u�333C��R>�����p������Q�C��{                                    Bx}�s  T          @�G�>�ff�������
=C���>�ff��{�/\)���C�
                                    Bx}���  �          @�=q>�(����׾������C�y�>�(���Q��,(�� �RC��=                                    Bx}��`  �          @��?\)��G��
=q����C�+�?\)��
=�3�
���C���                                    Bx}��  �          @�(�?���G��0����Q�C��?������<�����C��\                                    Bx}���  �          @�=q>�(���ff��  �6{C�xR>�(��z�H�K����C�q                                    Bx}��R  �          @��?5��{����;�C�)?5�y���Mp��(�C�0�                                    Bx}���  �          @���?G��������\)C�o\?G���{�/\)�
=C�B�                                    Bx}�ٞ  �          @�G�?8Q������  ���RC�U�?8Q��`  �aG��2Q�C��{                                    Bx}��D  �          @���?W
=��(�������
C�  ?W
=�U�fff�933C��                                    Bx}���  �          @���?����{����yG�C�` ?���a��W
=�)z�C�]q                                    Bx}��  T          @�G�?������\��
=C��
?���XQ��_\)�/�
C�|)                                    Bx}�6  T          @�G�?�33��p���  ����C��=?�33�\(��_\)�/�
C�=q                                    Bx}�"�  �          @�
=?B�\�:�H�r�\�N�C�T{?B�\�Q����R8RC�j=                                    Bx}�1�  �          @�{?�z��#�
�����\p�C�>�?�z������    C��                                     Bx}�@(  �          @�p�?��\��
��ff�k�HC��?��\���������
C�>�                                    Bx}�N�  �          @�ff?u�{��33�d�\C��=?u��\)����B�C���                                    Bx}�]t  �          @��R?(���b�\�Y���-=qC�E?(�ÿǮ�����C�p�                                    Bx}�l  �          @��?���n�R�AG���RC�7
?�Ϳ�z�����C�R                                    Bx}�z�  �          @���?:�H�W
=�j=q�;33C�)?:�H���
�����C�Ф                                    Bx}��f  �          @�Q�>�녿�p���G�C���>��?!G���ff¢��Bc��                                    Bx}��  T          @�{>�{�k���=q�C�8R>�{?���ffz�B�Ǯ                                    Bx}���  T          @�ff?��p�������C�b�?�?�33��ff�)B�Q�                                    Bx}��X  �          @�\)?��>�Q���33p�A��R?��@$z���(��`z�B�
=                                    Bx}���  �          @��?���>u��=q�@�z�?���@��|(��Q��BH=q                                    Bx}�Ҥ  �          @��R?�p��������
aHC��\?�p�?�����  �m  B;�H                                    Bx}��J  �          @�p�?�ff�+���ffB�C��)?�ff?�=q��
=�Ba��                                    Bx}���  �          @�?����ff���R�C�t{?��?��
��(��}33BX�R                                    Bx}���  �          @�p�?h�þ�
=��ff�fC�Ǯ?h��?�����B|�
                                    Bx}�<  T          @���>#�
��=q�����{C�Z�>#�
?��
���HB��                                    Bx}��  T          @�Q�?(���p�����#�C�%?(�?�����=q#�Bw\)                                    Bx}�*�  �          @��?��G����
�w�C���?�<#�
��p�§��?��\                                    Bx}�9.  �          @��?���n�R�1��z�C�'�?���G���{�}C���                                    Bx}�G�  �          @�ff���dz��[��/\)C�Ǯ����=q��{��C��H                                    Bx}�Vz  T          @��R>���C33�xQ��N��C�e>���h����33ffC���                                    Bx}�e   �          @��>aG��N�R�xQ��HffC���>aG���������C���                                    Bx}�s�  T          @�=q>u�Fff��  �Pz�C�AH>u�fff��\)��C��)                                    Bx}��l  �          @�Q�>�z��dz��^{�0Q�C�S3>�z�Ǯ��\)�)C�E                                    Bx}��  �          @���>���p���(����\C��3>��O\)�Vff�6�C��                                    Bx}���  �          @��R>����{�n{�2�\C���>���n�R�?\)���C��q                                    Bx}��^  �          @�Q�>�\)���H��p���
=C��R>�\)�XQ��[��5ffC�T{                                    Bx}��  �          @q�>����>{��\��\C��q>��������<(��g�C��f                                    Bx}�˪  
�          @��׾���  ��{���C��\���:=q�g��K
=C�]q                                    Bx}��P  �          @��þ8Q�����6ff�
�RC��q�8Q�����p��x��C�Ǯ                                    Bx}���  �          @���=��
�R�\�p���CQ�C��==��
��
=��33��C��R                                    Bx}���  �          @�33    �]p��mp��<G�C��    ������z��C��                                    Bx}�B  �          @�(����g
=�hQ��4C������\��(�W
C�|)                                    Bx}��  �          @�G��.{�������v(�C��.{���
����¯��CN)                                    Bx}�#�  �          @���(��:�H��G��\�HC���(��(����� ��C`�                                     Bx}�24  �          @������<����33�Y=qCz�H����(���
=\CRp�                                    Bx}�@�  �          @��H?fff�L����
=�C��
?fff@�R��
=�{��B�                                      Bx}�O�  �          @���>��p����G�C���>�?�ff�����=B��3                                    Bx}�^&  �          @�G�>�녿h����{(�C�/\>��?�=q������B��q                                    Bx}�l�  
o          @�Q�>\�����33�=C�q>\?������8RB��3                                    Bx}�{r  �          @�Q쾸Q쿐�����H�C|n��Q�?����G�G�B�\                                    Bx}��  �          @�G����׽�����
=C:����@G���(��u33B�33                                    Bx}���  q          @��Ϳ�(�=u��{Q�C1����(�@(����\�hffB�#�                                    Bx}��d  i          @�\)�h��<#�
���
�fC3��h��@
=��G��s��B�\                                    Bx}��
  "          @�G��k�    �p  �C4&f�k�?�(��U��n�\B�
=                                    Bx}�İ  
�          @��
@*�H��(�@���BW  C���@*�H�u�@G
=BC�o\                                    Bx}��V  T          @�{@*�H�@�ffBM��C�n@*�H���
@8Q�A��HC�~�                                    Bx}���  "          @�(�@(Q��2�\@��HB;(�C���@(Q����@ffA��
C��                                    Bx}��            @��@�
�7
=@q�B<�C��q@�
����@�\A�z�C��
                                    Bx}��H            @��\?�����{=���?��C�Ff?���������Q�C��
                                    Bx}��  
�          @���?�\)��Q�=L��?z�C�L�?�\)��
=�  ��=qC�f                                    Bx}��  T          @��\?˅�p  ?�\A��C��H?˅���;����33C���                                    Bx}�+:  
�          @�33@Q���@��BW\)C���@Q��qG�@+�A�(�C���                                    Bx}�9�  "          @�G�@  �#33@�Q�BF�
C��{@  ��33@��A���C�\)                                    Bx}�H�  �          @�p�?�z���
@��RBhC�� ?�z��aG�@;�B��C�E                                    Bx}�W,  �          @������@j=q@r�\B/�B�.����?\@���B��)C�                                    Bx}�e�  �          @�=q��ff@u@XQ�B(�B�aH��ff?�\)@�Q�B��fB���                                    Bx}�tx  �          @�p����@o\)@���B6�B�\���?�(�@�=qB��CQ�                                    Bx}��  �          @��Ϳ�ff@y��@z=qB0(�B�{��ff?�
=@�Q�B��RB��H                                    Bx}���  �          @����Q�@~{@~{B/  B�zῘQ�?��H@��HB�L�B��                                    Bx}��j  �          @�����@z=q@���B3Q�B�B����?У�@��
B��)B��                                    Bx}��  T          @�
=�G�@B�\@�  B\p�BиR�G�?&ff@�z�B��RC8R                                    Bx}���  �          @������@R�\@�{BOQ�B�aH���?�G�@�\)B��
B�W
                                    Bx}��\  �          @��
>�33@L��@��\B[�B�z�>�33?@  @�G�B��B�\                                    Bx}��  �          @���>�Q�@ff@�{B��B��\>�Q�#�
@�
=B��qC��                                    Bx}��  �          @�\)?��?aG�@�33B�ffB^�H?�����@�{B��C���                                    Bx}��N  T          @�  ?�\>�@���B�{Ak33?�\����@��\Bs�
C�3                                    Bx}��  
�          @�ff?�p����
@�G�B���C�˅?�p����@�
=BZQ�C��f                                    Bx}��  �          @��R@�\��ff@�\)B��HC��=@�\�(�@q�BG��C���                                    