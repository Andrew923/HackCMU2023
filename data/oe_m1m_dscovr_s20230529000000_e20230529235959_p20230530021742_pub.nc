CDF  �   
      time          *   Conventions       ACDD-1.3, Spase v2.2.3     title         /DSCOVR Magnetometer Level 2 One Minute Averages    id        Doe_m1m_dscovr_s20230529000000_e20230529235959_p20230530021742_pub.nc   naming_authority      gov.noaa.swpc      program       DSCOVR     summary       }Interplanetary magnetic field observations collected from magnetometer on DSCOVR satellite - 1-minute average of Level 1 data      keywords      _NumericalData.ObservedRegion.Heliosphere.NearEarth, NumericalData.MeasurementType.MagneticField    keywords_vocabulary       Spase v2.2.2   
references        �; DSCOVR TIME SERIES DATA AVERAGES ALGORITHM THEORETICAL BASIS DOCUMENT, v2.4; GSE TO GSM COORDINATE TRANSFORMATION ALGORITHM THEORETICAL BASIS DOCUMENT v2.1      metadata_link         �http://www.ngdc.noaa.gov/docucomp/page?xml=NOAA/NESDIS/NGDC/STP/Space_Weather/iso/xml/satellite-systems_dscovr.xml&view=getDataView&header=none    license       Spase.Access Rights.Open   institution       NOAA   source        DSCOVR Magnetometer Level 1    platform      'Deep Space Climate Observatory (DSCOVR)    
instrument        +boom-mounted triaxial fluxgate magnetometer    history       ,DSCOVR real-time telemetry processing system   	algorithm         FDSCOVR MAGNETOMETER LEVEL 1B DATA ALGORITHM THEORETICAL BASIS DOCUMENT     algorithmVersion      B      algorithmDate         
2015-10-15     processing_level      Level 2    processing_level_description      11-minute average using Hodges-Lehmann M-estimator      date_created      2023-05-30T02:17:42.800Z   date_calibration_data_updated         2023-05-09T00:00:00.000Z   time_coverage_duration        P01D   time_coverage_start       2023-05-29T00:00:00.000Z   time_coverage_end         2023-05-29T23:59:59.000Z   time_coverage_resolution      PT1M   creator_name      Doug Biesecker     creator_type      person     creator_institution       DOC/NOAA/NWS/NCEP/SWPC     creator_email         doug.biesecker@noaa.gov    creator_url       http://www.swpc.noaa.gov/      publisher_name         National Geophysical Data Center   publisher_type        institution    publisher_institution         DOC/NOAA/NESDIS/NGDC   publisher_email       william.rowland@noaa.gov   publisher_url          http://www.ngdc.noaa.gov/dscovr/   records_maximum         �   records_present         �   records_data        �   records_fill             records_missing                    time                description       "date and time for each observation     
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
short_name        overall_quality    C_format      %d     units         n/a    lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale        7lBx�L߀  �          @�{@'����H?Q�A{C�Q�@'���z�
=q�ȣ�C�+�                                    Bx�L�&  �          @��@R�\�_\)?xQ�A7\)C���@R�\�g��L���Q�C�!H                                    Bx�L��  
�          @�p�@@���tz�?#�
@�C�&f@@���tz��R��G�C�#�                                    Bx�Mr  T          @�{@8Q��s�
?�Q�A_�C��@8Q��\)���Ϳ�z�C���                                    Bx�M  
�          @��R@N�R�k�?E�A33C���@N�R�n�R��ff��
=C�p�                                    Bx�M(�  �          @��R@+�����?5A�C���@+�������R��Q�C��                                    Bx�M7d  "          @�@
=q���>��R@g�C�3@
=q���׿�\)�RffC�l�                                    Bx�MF
  �          @�p�@)���~�R?��\A@(�C���@)�����H���R�h��C�y�                                    Bx�MT�  T          @�p�@,(���G�?:�HA	�C��\@,(���=q�������C��q                                    Bx�McV  �          @�p�@����>W
=@p�C�8R@����Ϳ�
=�^�\C���                                    Bx�Mq�  �          @�p�@1G���Q�?�@�C�Q�@1G��~�R�=p��
�HC�j=                                    Bx�M��  
�          @�ff@Tz��I��?���A�p�C�Ff@Tz��c�
?8Q�AffC�~�                                    Bx�M�H  �          @�
=@I���^{?��A�ffC�q@I���tz�>�ff@�C�Ǯ                                    Bx�M��  �          @�
=@Z=q�\��?��
A@(�C�W
@Z=q�fff����33C���                                    Bx�M��  "          @�ff@J=q�k��@  ���C�L�@J=q�P  �������C�3                                    Bx�M�:  "          @�p�@Fff�Y��?�z�A��RC�/\@Fff�p  >��H@���C��=                                    Bx�M��  �          @�p�@?\)�O\)@(�A��C�XR@?\)�p��?��AD  C�7
                                    Bx�M؆  
�          @�p�@>{�P  @�A�33C�9�@>{�qG�?��AC�C��                                    Bx�M�,  
�          @�
=@333�Z�H@\)A���C���@333�}p�?��AAG�C���                                    Bx�M��  �          @�\)@G
=�XQ�?�
=A�{C�G�@G
=�tz�?B�\A�C��\                                    Bx�Nx  �          @�
=@8Q��Q�@�A��\C���@8Q��vff?�AZ�RC�b�                                    Bx�N  T          @�ff@�R��Q�?�
=A��C��H@�R��Q�=�Q�?��C��                                    Bx�N!�  �          @�\)@(����?�G�Am�C�S3@(��������Ϳ�z�C���                                    Bx�N0j  �          @��@�
��?���AU�C�p�@�
��=q��  �9��C��                                    Bx�N?  �          @�\)@/\)�r�\?�
=A��
C���@/\)���
>���@�=qC���                                    Bx�NM�  
�          @��@�
����?�\A�p�C��3@�
���
>�
=@���C��                                    Bx�N\\  �          @��@Q��|(�@33A��C��3@Q���z�?333@�\)C�AH                                    Bx�Nk  "          @��
@���S�
@AG�B�HC�=q@�����H?���A�G�C�q�                                    Bx�Ny�  
�          @��
@�R�K�@H��B��C���@�R��Q�?�p�A�z�C��                                    Bx�N�N  
�          @��H@'
=�C�
@J=qB��C�C�@'
=�x��@�\A�(�C��                                    Bx�N��  �          @���@1��S33@!G�A�RC�{@1��z�H?�{A}p�C���                                    Bx�N��  "          @���@$z��[�@ffA�{C�s3@$z��\)?�z�A[�C�g�                                    Bx�N�@  �          @���@�R�_\)@��A�RC�E@�R��=q?�Q�AeC�Z�                                    Bx�N��  T          @�\)@.�R�c33?�G�AK�C��=@.�R�l(��������C�B�                                    Bx�Nь  T          @�\)@=p��Mp�?У�A��C�S3@=p��c�
?
=q@�ffC��q                                    Bx�N�2  �          @��\@4z��\��?�=qA���C���@4z��vff?+�@�ffC�q                                    Bx�N��  T          @��
@N�R�Q�=�?�ffC�>�@N�R�J=q�h���5G�C���                                    Bx�N�~  T          @�z�@`  �R�\�O\)��C�e@`  �7��������C�O\                                    Bx�O$  �          @�(�@i���L(������
C�n@i���8Q쿹����{C�ٚ                                    Bx�O�  �          @��\@j=q�1�?���A�C�h�@j=q�E?�\@ÅC���                                    Bx�O)p  �          @��
@Q��E�?�ffA�z�C�l�@Q��^�R?B�\A�
C��                                    Bx�O8  �          @��
@N{�>{@ffA�33C���@N{�^�R?�=qAMG�C�c�                                    Bx�OF�  �          @�@dz��Dz�?��HAf�HC���@dz��R�\>L��@��C��\                                    Bx�OUb  T          @�=q@p  �R�\?fffA#
=C�c�@p  �Z=q�#�
��C��f                                    Bx�Od  �          @��@n{�U�?@  A	p�C��@n{�Y�����R�b�\C��f                                    Bx�Or�  �          @��@z=q�Mp�>���@mp�C�L�@z=q�I���.{��C���                                    Bx�O�T  
�          @�  @����녿Y����HC��{@��׿�z��G�����C��
                                    Bx�O��  �          @�Q�@�(��5=�\)?L��C���@�(��.{�O\)�(�C�K�                                    Bx�O��  �          @�Q�@�p��-p�>�p�@�\)C���@�p��,(�����G�C���                                    Bx�O�F  �          @�Q�@�(��Q���Ϳ�Q�C���@�(���R�Tz����C���                                    Bx�O��  �          @���@�Q��Q쿐���O�C�\)@�Q��  ������C�˅                                    Bx�Oʒ  
�          @�Q�@�녿�p��#�
��G�C�E@�녿�\)�#�
��p�C��
                                    Bx�O�8  T          @�  @�\)�
�H�����uC�
@�\)���H�z�H�5p�C�,�                                    Bx�O��  T          @���@�G��33����=qC�޸@�G���������Pz�C�B�                                    Bx�O��  �          @�
=@��\��33>#�
?��C�@��\���;�
=��(�C���                                    Bx�P*  
�          @�\)@�Q쿬�;k��)��C�3@�Q쿜(��!G��陚C��3                                    Bx�P�  
�          @��
@��H�˅���Ϳ���C�z�@��H��p��z����HC��                                    Bx�P"v  �          @�(�@��Ϳ��8Q���C�~�@��Ϳ�ff�����\C�.                                    Bx�P1  T          @�(�@��R��z��
=��z�C�
@��R�z�H�@  �G�C�"�                                    Bx�P?�  �          @���@�33���\(��!��C�p�@�33�����  �lQ�C�U�                                    Bx�PNh  T          @�p�@�p���{�@  ��C��H@�p����ÿ����Tz�C��\                                    Bx�P]  
�          @��@�=q�@  =��
?}p�C���@�=q�=p������(�C��                                     Bx�Pk�  
�          @�(�@��s33�5�\)C�C�@��0�׿u�8Q�C���                                    Bx�PzZ  T          @��@�33�   �����33C��@�33�#�
��\)����C��3                                    Bx�P�   �          @�z�@�\)�5����p�C�AH@�\)���
��\��C�޸                                    Bx�P��  �          @�(�@�\)�����Q���G�C�K�@�\)?�R�33����A��                                    Bx�P�L  T          @��
@��
>#�
�
=��(�@�\@��
?G���(���33A"=q                                    Bx�P��  T          @��@�
=�u�����{C��q@�
=?   �����@�{                                    Bx�PØ  T          @��
@�ff�#�
��33��G�C��3@�ff?\)�������\@�R                                    Bx�P�>  
�          @�(�@��\>\�����z�@�=q@��\?��\��p���{AT��                                    Bx�P��  �          @��
@��R>�녿����33@��@��R?s33��
=��{AAG�                                    Bx�P�  T          @�33@��>�33��ff��z�@�{@��?^�R��\)��A/\)                                    Bx�P�0  �          @�z�@��\>#�
����{?�(�@��\?&ff��ff���
A z�                                    Bx�Q�  �          @��R@��>��H��(����\@�Q�@��?xQ��  ����A>�R                                    Bx�Q|  
�          @��R@���>�녿�33����@�\)@���?u��Q�����A?�                                    Bx�Q*"  T          @�{@��?#�
�����
A@��?�
=��33��Q�Ak�                                    Bx�Q8�  �          @��H@�33?h�ÿ�����A<Q�@�33?����
��(�A���                                    Bx�QGn  �          @��
@�33?Tzῧ��{
=A#
=@�33?�
=��G��@(�Ae                                    Bx�QV  T          @�@�  ?8Q쿆ff�EG�A
=q@�  ?�  �J=q�  A>{                                    Bx�Qd�  "          @���@�ff?�z�J=q�ffA�  @�ff?�=q��{�z=qA�
=                                    Bx�Qs`  �          @�Q�@�=q?���W
=�{AC
=@�=q?��R���H���Ag
=                                    Bx�Q�  �          @���@�Q�?�33�.{���A��@�Q�?���u�.�RA���                                    Bx�Q��  T          @��R@�33?�G���(���ffA��@�33?�=�G�?��\A�                                      Bx�Q�R  T          @��R@��\?У׿^�R�"{A��R@��\?���33��33A���                                    Bx�Q��  �          @���@�p�>�33���H�t  @�Q�@�p�?0�׿�ff�T(�AG�                                    Bx�Q��  �          @�33@�  ��z����G�C�S3@�  ���H�^�R�(��C��                                     Bx�Q�D  �          @�33@�33��=q�L����C��
@�33��Q�8Q���C�^�                                    Bx�Q��  �          @��R@�p��  ��ff�H(�C��3@�p����Ϳ�z���  C�                                    Bx�Q�  �          @�{@����H>\@��C�S3@���G��u�:�HC��                                    Bx�Q�6  
�          @��@�(��Y��?xQ�A9��C���@�(����?.{A{C�`                                     Bx�R�  �          @��R@���#33>�z�@U�C�� @���!G������HC��                                     Bx�R�  T          @��R@�
=�#33>k�@*�HC�n@�
=� �׿���  C���                                    Bx�R#(  T          @�Q�@��� ��>�z�@U�C���@����R��ff���
C�f                                    Bx�R1�  �          @�Q�@���   >�\)@K�C��{@���{��ff��
=C��                                    Bx�R@t  �          @�Q�@�ff�.{=���?�
=C��@�ff�(Q�.{���\C��R                                    Bx�RO  �          @��@�33�5�=�Q�?�=qC���@�33�/\)�8Q��G�C�"�                                    Bx�R]�  T          @�\)@�=q�5��8Q���C��)@�=q�*=q�z�H�6�RC�o\                                    Bx�Rlf  �          @��@����333�B�\�Q�C��\@����p����
���HC�Y�                                    Bx�R{  T          @���@�p��'��L����\C��{@�p���Ϳp���/
=C���                                    Bx�R��  �          @���@�ff�'
=�:�H�ffC�\@�ff��\��Q���Q�C���                                    Bx�R�X  �          @���@~{�8�ÿz�H�4��C���@~{�\)��G�����C��3                                    Bx�R��  T          @�  @|���@�׿����=qC�XR@|���.�R������z�C���                                    Bx�R��  T          @��R@�=q�2�\��Q���{C�@�=q�%�����T��C��3                                    Bx�R�J  "          @���@����:�H��R��=qC�@����(Q쿵����C�p�                                    Bx�R��  �          @�  @}p��?\)�Ǯ��ffC�s3@}p��0�׿�(��b=qC���                                    Bx�R�  T          @�\)@e�S�
�k��)�C��@e�:�H����33C�o\                                    Bx�R�<  	�          @�\)@\)�:�H�����G�C��H@\)�,(���(��c�
C��                                    Bx�R��  
�          @�\)@z�H�@�׾����ffC�AH@z�H�0�׿�ff�r�\C�u�                                    Bx�S�  "          @��@j�H�Q�?��@�{C��@j�H�S�
��{�~�RC��
                                    Bx�S.  �          @�=q@�(��'
=?���ArffC��
@�(��8Q�?
=q@�33C��\                                    Bx�S*�  "          @��@}p��:=q?W
=A33C���@}p��A논#�
���
C�N                                    Bx�S9z  "          @���@z�H�7
=?�(�A`��C��{@z�H�E>�p�@��C���                                    Bx�SH   "          @�
=@p��� ��@�\A�(�C�)@p���>�R?�G�Ak
=C���                                    Bx�SV�  T          @�  @w
=�#�
?���A�C�7
@w
=�>{?��ADz�C�1�                                    Bx�Sel  �          @��H@���$z�?�(�A��C�f@���7�?333@��RC���                                    Bx�St  
�          @��@�  �.{?�{A�
=C��@�  �C�
?J=qA
=C�Q�                                    Bx�S��  �          @�(�@����!�?��RA`Q�C��
@����1G�>��H@�G�C���                                    Bx�S�^  �          @�(�@�33�333?��RA`z�C�޸@�33�A�>�
=@�
=C�Ǯ                                    Bx�S�  �          @�(�@��H��H?���AYG�C�t{@��H�)��>��H@��C�J=                                    Bx�S��  �          @�z�@���� ��?�
=AT��C�޸@����/\)>�G�@��RC��                                    Bx�S�P  �          @�{@�33�%?�33AL��C���@�33�333>���@��
C��R                                    Bx�S��  
�          @�
=@����?�  A]�C���@��,��?�@���C�Q�                                    Bx�Sڜ  �          @�ff@�(��   ?��\Ac
=C�*=@�(��0  ?
=q@�Q�C���                                    Bx�S�B  
�          @��@��R��R?��Ak�
C���@��R�   ?&ff@陚C�\)                                    Bx�S��  �          @��
@�
=���?�  A��C���@�
=�{?n{A(Q�C��                                    Bx�T�  
Z          @�G�@��   ?��An{C��R@���?333@��C�ff                                    Bx�T4  �          @�=q@�z����?�  Ad��C���@�z��p�?(�@�(�C�^�                                    Bx�T#�  "          @�=q@��H�ff?���AIp�C�˅@��H�#�
>�(�@��\C��q                                    Bx�T2�  
�          @�G�@�z��'�?�AV�\C���@�z��5>�
=@�  C��                                    Bx�TA&  �          @���@��
�-p�?}p�A6{C�U�@��
�8Q�>k�@$z�C��=                                    Bx�TO�  T          @��@�ff�$z�?�33AR�HC�@ @�ff�2�\>�
=@�  C�4{                                    Bx�T^r  �          @��\@�
=�!�?��RAaC���@�
=�1G�?�\@���C�Y�                                    Bx�Tm  
�          @��\@�=q��R?xQ�A0  C��@�=q�)��>�=q@A�C�9�                                    Bx�T{�  T          @�33@�  �-p�?333@�p�C���@�  �333�L�Ϳ��C�O\                                    Bx�T�d  �          @�(�@����,��?.{@�=qC��R@����1녽�\)�B�\C��{                                    Bx�T�
  �          @��@��
�'
=>��H@�G�C��
@��
�(�þaG��(�C�h�                                    Bx�T��  �          @��H@��H�(Q�>��R@\��C�j=@��H�'
=������G�C�y�                                    Bx�T�V  �          @�G�@����/\)?=p�A
=C�AH@����5���
�8Q�C��=                                    Bx�T��  �          @�=q@�{�1�>�G�@�  C�4{@�{�333���R�a�C�                                      Bx�TӢ  �          @���@����=q����BffC�^�@�����\��z����C�J=                                    Bx�T�H  "          @���@����#33��(���z�C��
@����ff��=q�H(�C��)                                    Bx�T��  T          @��@�{�$z����޸RC�:�@�{�zῠ  �j�HC�~�                                    Bx�T��  "          @���@��R�-p��B�\���C��H@��R�#�
�fff�$��C�W
                                    Bx�U:  �          @���@����'�>8Q�@�C�Q�@����%���\��G�C���                                    Bx�U�  �          @���@�\)�-p����
�B�\C��3@�\)�'
=�8Q��Q�C�,�                                    Bx�U+�  �          @���@��+��B�\�
�RC��{@���ÿ�
=��{C�'�                                    Bx�U:,  "          @�G�@��
��;�p���{C�Y�@��
�G��}p��6=qC�B�                                    Bx�UH�  "          @���@�����ý�G����RC�Z�@�����\�(�����C��                                    Bx�UWx  �          @��R@��H�zᾞ�R�hQ�C��\@��H�
=q�fff�(��C��                                    Bx�Uf  �          @�
=@����\��G�����C�Ff@�����333�C��{                                    Bx�Ut�  �          @�{@���G�>aG�@#�
C�=q@���  ��p����HC�W
                                    Bx�U�j  �          @��@���{>�\)@P  C�}q@���{�����b�\C��H                                    Bx�U�  �          @��
@��
�	��<�>���C��@��
������
C�8R                                    Bx�U��  �          @�p�@����
�H>8Q�@�C��)@����	���\��{C���                                    Bx�U�\  "          @���@�ff��\�#�
�   C��@�ff���H�\)���HC��                                    Bx�U�  "          @�z�@�\)��Q콸Q쿁G�C�K�@�\)���Ϳz�����C�                                    Bx�Ų  �          @��@�G���=q�k��(��C�@�G����H�.{� (�C��=                                    Bx�U�N  �          @�33@�Q��p���������C�~�@�Q���ÿTz����C�b�                                    Bx�U��  T          @�z�@��ÿ��
��p����
C�C�@��ÿУ׿O\)���C��                                    Bx�U��  "          @��@������>��?�\C���@����ff���R�mp�C�R                                    Bx�V@  �          @��H@��R����>8Q�@	��C��R@��R����z��\(�C�˅                                    Bx�V�  
�          @��
@�  ����>aG�@'
=C���@�  ���;�  �<(�C��
                                    Bx�V$�  �          @�z�@�\)���H>L��@33C�1�@�\)��Q쾙���aG�C�C�                                    Bx�V32  �          @��@�
=��z�=L��?z�C�l�@�
=��{��(���=qC��\                                    Bx�VA�  
�          @�z�@�\)��Q������C�Ff@�\)��׿���=qC��f                                    Bx�VP~  "          @���@�Q��zὸQ쿇�C���@�Q���ÿ\)�ҏ\C���                                    Bx�V_$  T          @�z�@��ÿ�=q<�>�p�C��)@��ÿ��
��
=��\)C�AH                                    Bx�Vm�  
�          @�z�@����z����
=C�� @�����ÿ
=��C��)                                    Bx�V|p  T          @��@���p�����p�C��3@���녿(���
=C�s3                                    Bx�V�  
Z          @��
@��R�����
�qG�C�W
@��R���
�G��(�C�R                                    Bx�V��  
Z          @��@�(��33�\��Q�C�t{@�(���33�^�R�%�C�N                                    Bx�V�b  �          @��H@��\�z�\���C�7
@��\���^�R�'
=C��                                    Bx�V�  �          @��H@~{�0�׾#�
���C��3@~{�(�ÿTz��C�,�                                    Bx�VŮ  "          @�33@��
�!녾�\)�S33C�:�@��
��ÿc�
�(��C��R                                    Bx�V�T  T          @��\@�������  C�>�@�������G�
C�>�                                    Bx�V��  
�          @�Q�@�=q�(������{C��f@�=q�G��}p��?�
C�n                                    Bx�V�  "          @��@����Q����{C�)@����(�����G\)C�R                                    Bx�W F  �          @��H@������W
=��
C���@��׿��Ϳ�=q�~=qC�H�                                    Bx�W�  "          @��
@�����ͿQ��  C�� @�����Ϳ��R�mG�C��                                    Bx�W�  
�          @�(�@�
=�z������C�� @�
=�ff����Y�C��H                                    Bx�W,8  
�          @�z�@dz��P��?\)@��C��{@dz��S33�k��.{C���                                    Bx�W:�  
�          @�(�@Y���[�?.{A z�C�e@Y���_\)�#�
��=qC�!H                                    Bx�WI�  �          @�(�@W��^�R?
=@���C��@W��aG�����C33C�ٚ                                    Bx�WX*  "          @���@[��Z�H?.{A Q�C��=@[��_\)����޸RC�E                                    Bx�Wf�  �          @���@]p��U?fffA)C�@]p��]p�=�Q�?�  C��                                    Bx�Wuv  �          @���@L(��fff?h��A+�C���@L(��mp�=#�
>�G�C�\)                                    Bx�W�  �          @�p�@N�R�h��?B�\AG�C��=@N�R�mp�����C��                                     Bx�W��  �          @��@R�\�g
=?�@ӅC�,�@R�\�h�þ����]p�C��                                    Bx�W�h  
�          @��@N{�g
=?Tz�A��C��@N{�l�ͼ��\C��f                                    Bx�W�  T          @�(�@HQ��hQ�?\(�A#�C�b�@HQ��n�R�#�
��Q�C���                                    Bx�W��  T          @�(�@C�
�j�H?xQ�A8  C��@C�
�r�\=�Q�?�=qC�t{                                    Bx�W�Z  
�          @�33@=p��k�?��AO�C�h�@=p��u�>L��@=qC��R                                    Bx�W�   
�          @���@AG��e?�
=A��C��@AG��tz�?\)@�=qC�*=                                    Bx�W�  
�          @�(�@4z��mp�?�p�A��\C�� @4z��|��?z�@��C���                                    Bx�W�L  �          @���@E��g
=?�A]�C�9�@E��q�>���@e�C��3                                    Bx�X�  �          @�p�@5�r�\?�G�A@Q�C�q�@5�z�H=�G�?�=qC���                                    Bx�X�  �          @�z�@��  ?��RA�(�C�+�@��\)?�@�Q�C�xR                                    Bx�X%>  �          @��
@
=�|(�?�\)A�
=C�y�@
=��ff?+�@�ffC��=                                    Bx�X3�  "          @�(�@   ���?�ffA���C�R@   ���?Tz�A��C�K�                                    Bx�XB�  "          @��@�\�{�@   A�ffC��
@�\����?�ffAH��C��)                                    Bx�XQ0  
�          @�33?����z�H@33A�z�C�7
?�������?�{AS�
C�<)                                    Bx�X_�  "          @��H?޸R�w�@ffA癚C�
?޸R����?�z�A���C�H                                    Bx�Xn|  �          @��
@ff�p��@G�Aޏ\C���@ff��?�\)A�z�C�T{                                    Bx�X}"  �          @�(�@�R�s33@ffAˮC�0�@�R��p�?�Q�Ab{C��                                    Bx�X��  
�          @�z�@ff�qG�@33A�{C��@ff��(�?�33AZffC�ٚ                                    Bx�X�n  
�          @��@p��i��@z�A�Q�C���@p���=q?���A���C�=q                                    Bx�X�  "          @��\@z��fff@p�AڸRC�b�@z���  ?�{A��HC��                                    Bx�X��  �          @�z�@z��u�?��A��RC���@z����?}p�A<  C���                                    Bx�X�`  �          @�z�?�z��g
=@5Bz�C�S3?�z�����?�p�A���C��H                                    Bx�X�  "          @�{?��e@=p�B  C���?�����@Aȏ\C���                                    Bx�X�  
�          @�p�?�
=�\��@E�B�C��q?�
=����@  A�33C�J=                                    Bx�X�R  "          @��?ٙ��\��@C33Bz�C�)?ٙ���G�@{A���C�k�                                    Bx�Y �  �          @�z�?��a�@:�HB(�C���?����H@�A��C��                                    Bx�Y�  �          @�p�?�Q��e@A�B=qC��?�Q���p�@�A�(�C���                                    Bx�YD  �          @�ff@   �{�@  A�  C�y�@   ��=q?��A{�C�ff                                    Bx�Y,�  �          @�{@��[�@"�\A�
=C���@��y��?޸RA�C�                                      Bx�Y;�  �          @���@!G��O\)@9��B  C���@!G��r�\@��A�
=C��{                                    Bx�YJ6  �          @�G�@>{�L��@\)A�p�C�p�@>{�j=q?޸RA�Q�C���                                    Bx�YX�  �          @��\@+��U@.�RB33C�\)@+��vff?���A���C�j=                                    Bx�Yg�  �          @��@!G��Vff@4z�B�C�u�@!G��xQ�@�\A��HC��                                     Bx�Yv(  "          @��H@0  �hQ�@  A�ffC��
@0  ����?�
=A�z�C�.                                    Bx�Y��  �          @�33@(���o\)@�RA�C���@(�����
?���A|��C�S3                                    Bx�Y�t  
�          @��
@6ff�e�@�\A��HC�=q@6ff�\)?�p�A��RC��                                    Bx�Y�  �          @�z�@;��i��@
=A�ffC�` @;���Q�?��AiC��                                    Bx�Y��  T          @�(�@H���dz�?�
=A���C��=@H���y��?��ALz�C�o\                                    Bx�Y�f  �          @�(�@J=q�fff?���A�z�C���@J=q�z=q?��
A8��C��H                                    Bx�Y�  �          @�{@J=q�i��?�A��C�t{@J=q�~{?�{AFffC�E                                    Bx�Yܲ  �          @�ff@J�H�hQ�?���A�(�C���@J�H�}p�?�33AL��C�Y�                                    Bx�Y�X  F          @�p�@Fff�j�H?�33A�ffC��@Fff�\)?��AC�C��=                                    Bx�Y��  �          @��
@>{�u?�=qA�(�C��)@>{���\?@  A\)C��                                    Bx�Z�  T          @�(�@Fff�qG�?���A�z�C��R@Fff��Q�?B�\A  C�ٚ                                    Bx�ZJ            @���@HQ��r�\?\A��C�@HQ�����?5@�(�C��\                                    Bx�Z%�  �          @�@G
=�xQ�?�(�A�(�C�\)@G
=��33?&ff@�{C��R                                    Bx�Z4�  "          @�p�@Fff�~�R?�{AFffC��3@Fff��(�>�=q@A�C�w
                                    Bx�ZC<  �          @�@G��}p�?�
=ARffC�!H@G����>�33@w
=C��
                                    Bx�ZQ�  �          @�p�@AG��|��?�{AuG�C��{@AG���z�?�@�ffC��                                    Bx�Z`�  T          @�{@Dz��w�?�ffA���C�8R@Dz����?=p�A\)C�g�                                    Bx�Zo.  "          @�@Fff�~�R?�33AMG�C���@Fff��(�>���@fffC�w
                                    Bx�Z}�  "          @�ff@HQ�����?n{A$��C�ٚ@HQ�����=���?�=qC��                                     Bx�Z�z  T          @�p�@G��x��?�{As�
C�\)@G����\?��@���C��\                                    Bx�Z�   "          @�{@L���q�?�ffA��HC�%@L����Q�?E�A�C�N                                    Bx�Z��  
�          @�
=@Z�H�n{?��Ag�C�Ff@Z�H�z=q?
=q@�C���                                    Bx�Z�l  "          @��R@Vff�s�
?�z�AM�C��@Vff�}p�>\@��RC��                                    Bx�Z�  �          @�{@P  �tz�?���Ak�
C�+�@P  ��Q�?
=q@�
=C��H                                    Bx�Zո  �          @�p�@G
=�u�?�  A�C���@G
=����?8Q�A�C���                                    Bx�Z�^  "          @���@HQ��w�?��Ak�C�u�@HQ�����?�@�33C��3                                    Bx�Z�  T          @���@AG��w�?��
A�z�C��
@AG���33?@  Ap�C�0�                                    Bx�[�  
�          @�(�@AG��w
=?�p�A��C�  @AG����\?333@��C�B�                                    Bx�[P  T          @��
@��~{@
�HAɮC�=q@����?�{Aw�C�8R                                    Bx�[�  
�          @�(�@ ���z�H@	��A��
C�U�@ ����Q�?���As�
C�E                                    Bx�[-�  �          @�z�@(Q���(�?�G�A��RC�AH@(Q���33?.{@�p�C��q                                    Bx�[<B  "          @��H@7���  ?�p�A`��C��)@7����>�G�@�  C�S3                                    Bx�[J�  
Z          @��@?\)�|��?��A=�C���@?\)��=q>�=q@B�\C��                                    Bx�[Y�  T          @��
@H���|��?\(�A�C�:�@H����G�=���?���C���                                    Bx�[h4  T          @�33@aG��l(�=���?���C��\@aG��h�ÿ����Q�C���                                    Bx�[v�  "          @�=q@L(��z�H>�{@u�C��3@L(��z=q������C��R                                    Bx�[��  "          @�z�@Y���u�>�Q�@���C�Ǯ@Y���u���Q���=qC�Ǯ                                    Bx�[�&  
�          @�(�@Q��{�>\@�ffC��@Q��{���p���33C��\                                    Bx�[��  "          @���@L(��u?!G�@��C�ٚ@L(��x�ý��Ϳ�33C���                                    Bx�[�r  
�          @���@J�H�vff?��@�G�C��q@J�H�xQ�8Q��   C��)                                    Bx�[�  �          @���@S�
�p  ?��@�z�C��
@S�
�s33���Ϳ�33C��=                                    Bx�[ξ  
�          @���@W��o\)>k�@(��C�f@W��n{����Q�C��                                    Bx�[�d  "          @�G�@XQ��i��?:�HAffC�g�@XQ��n�R=L��?z�C�                                      Bx�[�
  �          @��\@c�
�e�?
=@�(�C�p�@c�
�g����
�aG�C�AH                                    Bx�[��            @��H@a��g�?
=q@�(�C�&f@a��j=q����C���                                    Bx�\	V  b          @��@^{�j�H>L��@\)C���@^{�i��������C��                                    Bx�\�  
�          @���@N{�qG�?�\@��C�>�@N{�s33�8Q��C�!H                                    Bx�\&�  
(          @���@<(���  ?Tz�A�
C�'�@<(����H=���?�=qC�޸                                    Bx�\5H  
�          @�G�@;��{�?�33AR�RC�\)@;����\>�
=@�=qC��                                     Bx�\C�  �          @���@4z��|��?�G�Ai��C���@4z����?
=q@���C�9�                                    Bx�\R�  T          @���@����33?У�A���C�q@����=q?^�RA Q�C�u�                                    Bx�\a:  
�          @���@p�����?ٙ�A�ffC���@p�����?uA0(�C��q                                    Bx�\o�  
�          @�  @2�\�u�?��
A�33C�@2�\��G�?Tz�Ap�C�T{                                    Bx�\~�  "          @���@G��k�?�{A{�C�  @G��w
=?0��@��C�s3                                    Bx�\�,  "          @�{@Mp��mp�>���@z=qC�p�@Mp��mp����
�o\)C�o\                                    Bx�\��  T          @�\)@����33�{�ԸRC��@�����R�{��C��R                                    Bx�\�x  "          @�  @�=q��{�p���Q�C�@�=q��Q��   ��ffC�@                                     Bx�\�  
�          @���@����Q�����
=C���@����ff�����C���                                    Bx�\��  �          @��H@�Q��
=���R����C�˅@�Q�Ǯ�33����C��                                    Bx�\�j  
Z          @�z�@��
��z��
=�£�C��{@��
���\�����
C��q                                    Bx�\�  
�          @�z�@�p���(��������C�U�@�p���\)�{��33C�h�                                    Bx�\�  T          @�=q@����!녿����{C�H@����  ��ff��  C�q�                                    Bx�]\  
�          @���@~{�:=q�z�H�4��C��q@~{�,�Ϳ��H����C�޸                                    Bx�]  T          @��@z�H�7����
�>�RC��@z�H�*=q��  ��=qC��                                    Bx�]�  �          @��@\)�#�
���R�jffC��f@\)��
��33����C��                                    Bx�].N  �          @�{@�z�����\)�Qp�C��f@�z��p���G����\C��                                    Bx�]<�  
�          @�ff@��
�#33�}p��8(�C�q@��
�ff��33��\)C�#�                                    Bx�]K�  �          @�  @��R�
=���
�mp�C�XR@��R����33���RC��                                    Bx�]Z@  
Z          @�\)@�=q�.{�^�R�!�C�%@�=q�"�\����s�C��                                    Bx�]h�  	�          @�Q�@��\�1녿W
=��C���@��\�'
=���
�m��C���                                    Bx�]w�  	�          @�\)@����6ff�.{���C�\)@����,�Ϳ����R�RC��                                    Bx�]�2  
(          @�  @����5�5��\C���@����+���z��W33C�AH                                    Bx�]��  
�          @��@x���C�
��R��C�� @x���:�H��{�N=qC���                                    Bx�]�~  
�          @���@~{�@�׿����C�q�@~{�8Q쿁G��9C�f                                    Bx�]�$  "          @�Q�@}p��@  ���љ�C�u�@}p��7���ff�A�C��                                    Bx�]��  T          @���@tz��C33��  �>�RC���@tz��>{�=p��33C��                                    Bx�]�p  �          @�{@xQ��AG���ff����C��@xQ��:=q�n{�-�C���                                    Bx�]�  
(          @�@y���?\)�����G�C�=q@y���;��!G��陚C���                                    Bx�]�  
�          @��@x���@  ��\)�^�RC�/\@x���<�Ϳ������C�j=                                    Bx�]�b  T          @�@x���A녾.{���HC�
=@x���=p��#�
��  C�U�                                    Bx�^
  
�          @�@x���@��=#�
>��C�"�@x���>�R��G���33C�G�                                    Bx�^�  T          @��R@x���B�\�����s33C���@x���=p��L�����C�^�                                    Bx�^'T  "          @���@u�K����H���C�8R@u�C�
�z�H�4  C��)                                    Bx�^5�  "          @�Q�@q��Mp���R���
C��3@q��Dz῎{�M�C�n                                    Bx�^D�  
�          @�\)@qG��L(��������C���@qG��C�
��=q�H��C�y�                                    Bx�^SF  
�          @�  @vff�I���Ǯ����C�Z�@vff�C33�^�R� ��C�˅                                    Bx�^a�  
(          @�\)@\)�=p���=q�EC���@\)�8�ÿ8Q��z�C�\                                    Bx�^p�  
�          @�@xQ��A녾k��-p�C�H@xQ��=p��0��� ��C�T{                                    Bx�^8  	�          @��R@�  �:�H�8Q��33C��@�  �7
=��R��z�C�9�                                    Bx�^��  	�          @�\)@�{�+����
���C��3@�{�)����(�����C��q                                    Bx�^��  "          @��@����<(�����(�C��3@����8Q����Q�C�1�                                    Bx�^�*  
(          @�
=@vff�HQ쾔z��S�
C�u�@vff�C33�@  �
�HC��\                                    Bx�^��  
�          @��@|(��A녾�\)�O\)C�7
@|(��=p��:�H�ffC���                                    Bx�^�v  �          @�  @����<(����
�n�RC��@����7
=�@  �33C�G�                                    Bx�^�  
�          @�\)@�G��9������{C�#�@�G��7�����  C�O\                                    Bx�^��  
Z          @�
=@���� �׾L���C��f@�����Ϳ���=qC�                                    Bx�^�h  
�          @�=q@���(�þ����VffC�=q@���$z�.{��C���                                    Bx�_  
�          @��@�33�#�
��  �5�C��f@�33�\)��R���C��                                    Bx�_�  	�          @�=q@�=q�)���u�,(�C�:�@�=q�%��R��\)C���                                    Bx�_ Z  
�          @���@��
��R�#�
��ffC�5�@��
��;Ǯ��  C�]q                                    Bx�_/   �          @���@�����Q�?W
=A�\C�k�@�����\?\)@˅C���                                    Bx�_=�  
�          @�{@�G��?!G�@�(�C��{@�G����>�z�@[�C�]q                                    Bx�_LL  �          @�
=@\)�7�?!G�@�  C�)@\)�;�>W
=@{C��{                                    Bx�_Z�  T          @�(�@p  �C�
>L��@z�C�aH@p  �C�
�u�1�C�e                                    Bx�_i�  
Z          @���@w��9���E��z�C��
@w��0�׿��\Q�C�>�                                    Bx�_x>  �          @�=q@�33�5��fff�#
=C��{@�33�+����
�jffC�p�                                    Bx�_��  "          @���@����*=q�p���,Q�C��@���� �׿�ff�o�C�k�                                    Bx�_��  T          @��H@�\)�%�����N�\C�T{@�\)�����p���\)C�<)                                    Bx�_�0  
�          @�G�@����!녿B�\�
{C���@����=q�����I�C�T{                                    Bx�_��  
�          @��@��R�{�z�H�5�C��f@��R�zῧ��s
=C��3                                    Bx�_�|  T          @�Q�@���{�����ffC��@���Q�L���Q�C��=                                    Bx�_�"  �          @�
=@����H�������\C�Y�@����8Q��G�C��)                                    Bx�_��  �          @�{@�G���þ\��p�C�q�@�G��z�333�{C���                                    Bx�_�n  T          @��R@������0���G�C���@����  ��  �;�C��                                    Bx�_�  
�          @��R@����H��33���HC�U�@���
=�+���G�C��                                    Bx�`
�  �          @�  @{��B�\<#�
>�C�"�@{��AG��Ǯ��  C�@                                     Bx�``  
�          @��\@s�
�P�׼���z�C��
@s�
�N�R��ff��C���                                    Bx�`(  "          @�G�@a��dz�>�@�
=C�U�@a��fff<#�
=�C�4{                                    Bx�`6�  �          @��@o\)�]p�������C��H@o\)�Z�H�����
C�Ф                                    Bx�`ER  �          @�p�@qG��_\)���
�`��C���@qG��Z�H�B�\�Q�C���                                    Bx�`S�  T          @�(�@n{�\(�>�@�{C��)@n{�^{=#�
>��C�xR                                    Bx�`b�  
�          @�(�@dz��fff?&ff@�Q�C�c�@dz��i��>8Q�@�C�.                                    Bx�`qD  �          @��@Y���qG��W
=���C�  @Y���n{�0����G�C�8R                                    Bx�`�  T          @��
@l(��a논#�
��C�#�@l(��`  ����(�C�B�                                    Bx�`��  "          @��@o\)�^{>.{?��HC���@o\)�]p�����<(�C��                                     Bx�`�6  
�          @�33@�  �E�?:�HA�C�5�@�  �H��>�{@s�
C���                                    Bx�`��  
�          @��
@�Q��AG�?O\)A�C�~�@�Q��Fff>�(�@�z�C�%                                    Bx�`��  
�          @��@�(��!G�>���@�\)C��@�(��#33=��
?k�C��                                    Bx�`�(  "          @��\@����   ��G���Q�C�7
@����p��������C�^�                                    Bx�`��            @���@���Q�u�,(�C�xR@���   ���H�ZffC�(�                                    Bx�`�t  b          @��R@z=q�E��L����\C��=@z=q�=p����V�RC�w
                                    Bx�`�  T          @�\)@Z�H�|��=�G�?�
=C�xR@Z�H�{���p�����C��f                                    Bx�a�  �          @�
=@=p�����?   @�\)C�\)@=p��������
�B�\C�C�                                    Bx�af  �          @�p�@$z����?B�\A33C�%@$z���
=>k�@!G�C���                                    Bx�a!  T          @�p�@  ��G�?h��A"ffC�.@  ���
>�Q�@�G�C��q                                    Bx�a/�  "          @�{@�����?��
Adz�C��@���z�?=p�A�RC���                                    Bx�a>X  "          @�@G����?�z�A}�C���@G���{?^�RAffC���                                    Bx�aL�  �          @�(�?�������?���AzffC��=?������?W
=A�
C�^�                                    Bx�a[�  "          @���@3�
���?E�A
ffC��@3�
����>�\)@FffC���                                    Bx�ajJ  	�          @�z�@Q���{?�  A4  C�
@Q�����>�@��C��)                                    Bx�ax�  �          @��
@{��p�?B�\A	p�C���@{��
=>�  @6ffC�u�                                    Bx�a��  "          @��
@!����?Q�AQ�C��@!���p�>��
@g�C��f                                    Bx�a�<  �          @�z�@.{��=q?�@�C�!H@.{��33=#�
>�(�C��                                    Bx�a��  �          @��@E���>���@h��C�w
@E���
�\)�\C�n                                    Bx�a��  "          @�p�@<�����R>�@��C���@<�����    =L��C�l�                                    Bx�a�.  
�          @�p�@7
=����?
=q@���C�޸@7
=���=�\)?8Q�C���                                    Bx�a��  T          @�p�@#33��{?0��@�p�C���@#33���>L��@
�HC��\                                    Bx�a�z  "          @���@���ff?333@��C�Q�@���  >W
=@�C�0�                                    Bx�a�   
�          @��R@1����>�(�@��RC�K�@1���(��L�Ϳ\)C�<)                                    Bx�a��  �          @�(�@p����\?!G�@��C��H@p����
>�?��HC�Ǯ                                    Bx�bl  �          @��H@{��G�?
=q@���C�H@{��=q=L��?��C���                                    Bx�b  
(          @�=q@(�����>�=q@C�
C���@(����=q�B�\��C��
                                    Bx�b(�  
�          @�33@(Q����=#�
>��HC��3@(Q����H��
=��
=C��H                                    Bx�b7^  T          @�33@,(����\����p�C��@,(��������H��Q�C��)                                    Bx�bF  	�          @�33@1���Q�=���?�{C���@1���  ��33�{�C��3                                    Bx�bT�  �          @�=q@������#�
��\C��=@����(��   ��
=C���                                    Bx�bcP  �          @�33@0������=�\)?=p�C�t{@0����  �\��  C��H                                    Bx�bq�  �          @��H@7���(�<#�
=�\)C�^�@7������
=����C�p�                                    Bx�b��  T          @�(�@{���R�����i��C�xR@{����G��Q�C��                                     Bx�b�B  �          @�z�@'�������
�W
=C�Y�@'���(�������RC�k�                                    Bx�b��  T          @��H@HQ��}p�>�Q�@�33C�+�@HQ��~�R�#�
��G�C�q                                    Bx�b��  T          @���@Dz��\)>�=q@AG�C��@Dz���  �\)���C���                                    Bx�b�4  T          @��H@Q��w�>u@*�HC�q@Q��xQ�����Q�C��                                    Bx�b��  �          @��H@L(��}p��L�Ϳ
=qC�g�@L(��|(���G���ffC�}q                                    Bx�b؀  �          @�=q@n{�W
=���R�`  C��
@n{�S�
�!G���Q�C�*=                                    Bx�b�&  �          @��\@~{�I����G����HC��\@~{�G������p�C���                                    Bx�b��  �          @��@��H�*=q���Ϳ�{C�=q@��H�(�þ�Q���  C�XR                                    Bx�cr  �          @��
@���*=q=�Q�?xQ�C�L�@���*=q�.{��z�C�P�                                    Bx�c  T          @��@��
�)��    ���
C�c�@��
�(Q쾀  �6ffC�q�                                    Bx�c!�  
�          @�Q�@�p��0�׾���p�C�@ @�p��/\)�Ǯ��
=C�]q                                    Bx�c0d  �          @��R@��H�1G�>�=q@J�HC��3@��H�1�<#�
>\)C��                                    Bx�c?
  T          @�
=@�p��%�?=p�A	�C�*=@�p��(Q�?   @���C��                                     Bx�cM�  �          @�\)@��\�2�\?�@�z�C��\@��\�5�>�=q@G�C��H                                    Bx�c\V  �          @��R@�33�&ff?E�A�C�˅@�33�*=q?�@��
C��                                     Bx�cj�  
(          @�  @���)��?c�
A&=qC�u�@���.{?#�
@�  C�q                                    Bx�cy�  
�          @��?�33��{?
=@�ffC��?�33��\)>.{?�\)C���                                    Bx�c�H  /          @��@�p����H?���Aj=qC�g�@�p���
?��\AFffC��R                                    Bx�c��  T          @�z�@�녿���?�{ARffC��H@�녿�Q�?}p�A:ffC�7
                                    Bx�c��  
�          @�{@��H��z�?���AS\)C�n@��H��G�?�  A:�\C��                                    Bx�c�:  "          @��R@����?aG�A$��C��@���ff?.{@�z�C���                                    Bx�c��  
�          @�@�=q�(Q�?Tz�A
=C��\@�=q�,(�?
=@�ffC�AH                                    Bx�cц  �          @��@g
=�L��?E�AG�C�9�@g
=�P  >��H@�Q�C���                                    Bx�c�,  
�          @��
@P  �W�?���AX��C��)@P  �\��?Tz�A\)C��H                                    Bx�c��  T          @�(�@Q��Tz�?���AN�HC�O\@Q��Y��?G�A�RC��R                                    Bx�c�x  T          @�ff@tz��?\)?Y��A
=C��q@tz��C33?
=@�p�C���                                    Bx�d  �          @�
=@����>�Q�@�C�˅@��{>.{@G�C���                                    Bx�d�  "          @�{@����Q�>Ǯ@��HC���@����=q>B�\@��C�c�                                    Bx�d)j  
�          @�p�@���
=?+�@���C�p�@���=q>�@�=qC�1�                                    Bx�d8  
�          @���@�G���Q�?z�@ڏ\C�Ф@�G���p�>�G�@�p�C���                                    Bx�dF�  
�          @�@�(����?+�@���C��@�(��˅?
=q@ə�C���                                    Bx�dU\  T          @��@�
=���H?(��@���C�c�@�
=��G�?
=q@�  C��                                    Bx�dd  �          @��@�=q��z�?��@ʏ\C���@�=q����>Ǯ@�G�C�xR                                    Bx�dr�  �          @���@��\��p�>�ff@�33C�Ff@��\� ��>�z�@QG�C��                                    Bx�d�N  T          @��@�p���
=��z��W
=C�q@�p���33��
=��(�C�E                                    Bx�d��  "          @��R@��ff?�R@�  C�Y�@����>�@��C�                                      Bx�d��  T          @�Q�@���(�?W
=A\)C�C�@����
?8Q�Ap�C��                                    Bx�d�@  
�          @��@���2�\?�@���C���@���5�>�Q�@��
C��                                    Bx�d��  
�          @�\)@�����?B�\A��C��@����\)?
=@�=qC���                                    Bx�dʌ  �          @�\)@�33��\?=p�A  C��@�33�?\)@�ffC�ٚ                                    Bx�d�2  y          @�@�Q��0��?@  Az�C��q@�Q��3�
?
=q@�=qC��H                                    Bx�d��  
�          @�@��H�+�?�@���C�^�@��H�.{>�p�@�G�C�33                                    Bx�d�~  y          @��R@���,(�?+�@���C�^�@���/\)>��@�p�C�*=                                    Bx�e$  
�          @�@�33�'
=?L��A�\C�� @�33�*=q?(�@�(�C�~�                                    Bx�e�  
�          @��@����?5A Q�C�=q@���{?��@ƸRC�H                                    Bx�e"p  �          @�(�@��
�
�H>�G�@�p�C�o\@��
�(�>�\)@I��C�L�                                    Bx�e1  
�          @��@���33>��@;�C���@���z�=���?��C��=                                    Bx�e?�  "          @�
=@�p�������z�C��H@�p��z�B�\�ffC��=                                    Bx�eNb  
�          @��H@��׿��#�
���C�
@��׿�ff�.{��\C�!H                                    Bx�e]  �          @�G�@�  �ٙ�=#�
>�C���@�  �ٙ���\)�L��C���                                    Bx�ek�  
�          @�Q�@�(���=�Q�?�G�C�>�@�(���
=�#�
��G�C�<)                                    Bx�ezT  
�          @���@�����R>�?�C�}q@�����R���
�uC�xR                                    Bx�e��  T          @��
@����=�G�?��C�E@���ý#�
���C�AH                                    Bx�e��  
(          @�z�@��
��
=��
?c�
C�K�@��
��
��\)�E�C�K�                                    Bx�e�F  T          @��@�{�(�>W
=@��C��@�{���=�\)?333C��)                                    Bx�e��  T          @��@�
=�
=>k�@\)C�|)@�
=��=�Q�?�  C�o\                                    Bx�eÒ  �          @�@�Q���>��
@X��C���@�Q��ff>8Q�?�Q�C���                                    Bx�e�8  �          @�{@�����\>���@a�C��\@�����
>L��@�C�ٚ                                    Bx�e��  "          @��@�����>L��@�C�8R@����H=��
?Q�C�,�                                    Bx�e�  
�          @�=q@�33���>�33@n�RC��@�33�
=q>W
=@  C���                                    Bx�e�*  T          @��H@����\>�33@o\)C�B�@����
>L��@�C�,�                                    Bx�f�  �          @��
@����>8Q�?�Q�C�u�@����=#�
>��C�l�                                    Bx�fv  
�          @��H@������=���?��C���@�����ýL�Ϳ��C��                                    Bx�f*  
�          @�(�@���!G�=���?���C�  @���!G��L�Ϳz�C���                                    Bx�f8�  "          @�(�@��R�&ff=u?
=C���@��R�&ff���Ϳ���C��=                                    Bx�fGh  "          @��H@��
�*�H>8Q�?��HC���@��
�+�<��
>k�C��
                                    Bx�fV  "          @�=q@�G��0��>�  @*=qC�Z�@�G��1G�=��
?aG�C�O\                                    Bx�fd�  "          @�z�@�{�'�>�Q�@w
=C�k�@�{�(��>W
=@p�C�W
                                    Bx�fsZ  �          @�z�@�{�(Q�>�p�@|��C�]q@�{�)��>aG�@�
C�H�                                    Bx�f�   
�          @��
@��'�>�z�@C33C�Y�@��(��>�?�
=C�K�                                    Bx�f��  
�          @��
@���!G�>B�\?��RC��q@���!�=#�
>�ffC���                                    Bx�f�L  "          @�33@����
=>.{?�\)C���@�����=#�
>�(�C��q                                    Bx�f��  
�          @��
@��H��
>#�
?�  C�:�@��H�z�<�>�p�C�4{                                    Bx�f��  
Z          @�=q@���=q>#�
?�C���@���=q<��
>k�C��H                                    Bx�f�>  T          @��H@�Q����>�Q�@z=qC���@�Q��=q>k�@\)C��\                                    Bx�f��  
�          @��
@������>�@�ffC���@�����H>���@c33C���                                    Bx�f�  
          @��@�ff�1�?Q�A�C���@�ff�4z�?(��@�C�Ф                                    Bx�f�0  	m          @��H@z�H�W�?��Ab�\C��@z�H�\(�?�\)AB�\C�`                                     Bx�g�  "          @�=q@s33�\(�?��As\)C���@s33�`��?��HAR�RC���                                    Bx�g|  �          @�G�@j�H�a�?�Ax��C�
=@j�H�fff?�p�AW�C���                                    Bx�g#"  
�          @���@y���Vff?��Aip�C���@y���Z�H?�z�AJffC�b�                                    Bx�g1�  
�          @�33@����:�H?�{Ak
=C�ٚ@����>�R?��HAPz�C���                                    Bx�g@n  "          @�=q@�z��(Q�?�
=Ax��C���@�z��,��?��A`��C�7
                                    Bx�gO  T          @���@�
=�z�?��A�C�L�@�
=���?�Az=qC��                                    Bx�g]�  "          @���@�33��H?޸RA�Q�C�|)@�33�   ?�\)A�33C��                                    Bx�gl`  
M          @�\)@��R� ��?���A���C���@��R�&ff?�Q�A�\)C�(�                                    Bx�g{  
�          @���@�(��,(�?�33A�z�C�y�@�(��1�?�\A�Q�C��                                    Bx�g��  	�          @���@�33�-p�?�A�ffC�C�@�33�333?��A�=qC��
                                    Bx�g�R  T          @�Q�@|���8��?�
=A�{C��@|���>�R?��A�33C��                                    Bx�g��  �          @�\)@l���C33@G�A�Q�C�B�@l���H��?�\)A���C�ٚ                                    Bx�g��  T          @���@S33�dz�@33A���C�aH@S33�j=q?��A�33C�f                                    Bx�g�D  
�          @�  @>{�w
=?�=qA�Q�C��=@>{�|(�?�33A��C���                                    Bx�g��  
�          @�  @H���vff?�Q�A�ffC��R@H���z�H?�G�A�Q�C�U�                                    Bx�g�  
g          @��R@Dz��g
=@��A��C�1�@Dz��mp�?�(�A�Q�C��
                                    Bx�g�6  	�          @��R@Z�H�Dz�@��A�(�C��@Z�H�K�@��AΣ�C���                                    Bx�g��  �          @��R@I���X��@=qA�C�t{@I���_\)@��A��C��                                    Bx�h�  �          @�@C33�_\)@33A�G�C��{@C33�e�@	��A�ffC�33                                    Bx�h(  
          @�G�@333�c33@(�A��HC�!H@333�h��@�\A���C�Ǯ                                    Bx�h*�  
5          @���@1G��q�@
=A�=qC�"�@1G��w
=?���A��RC��{                                    Bx�h9t  
Z          @�=q@<���~�R?��HA��RC�E@<������?��A��C��                                    Bx�hH  
�          @�=q@:�H����?��A��\C�H@:�H���H?�(�A�\)C��H                                    Bx�hV�  "          @��H@>�R��  ?�z�A�C�Y�@>�R��=q?�  A���C��                                    Bx�hef  "          @�33@=p�����?�=qA��\C��@=p����
?�A��
C�ٚ                                    Bx�ht  T          @��
@7
=��33?�A�Q�C�xR@7
=��p�?�G�A���C�<)                                    Bx�h��  
�          @�z�@0�����\@�A��HC��@0�����@G�A�(�C��f                                    Bx�h�X  
�          @�z�@+���
=?��RA�  C�4{@+���G�?�=qA�33C���                                    Bx�h��  �          @�{@�����R?�z�A�=qC�ff@������?޸RA��C�7
                                    Bx�h��  �          @�p�@���ff?�A��
C�U�@�����?�  A��HC�&f                                    Bx�h�J  T          @���@����33?�G�A�Q�C��=@�����?˅A��C���                                    Bx�h��  "          @��
@Q����
?�
=A�(�C�c�@Q���?�G�A��C�AH                                    Bx�hږ  
�          @�Q�?�����ff?���Ah��C�E?�����  ?�33AJffC�,�                                    Bx�h�<  �          @��@   ���?�Q�A��HC���@   ��
=?�ffA�
=C�O\                                    Bx�h��  T          @��R@   ���
?���A�{C�G�@   ��?�A���C�!H                                    Bx�i�  �          @�\)?L�����H?   @���C�u�?L�����>���@g�C�s3                                    Bx�i.  �          @��R?ٙ���Q�?��HAV�RC�˅?ٙ�����?�ffA9C��R                                    Bx�i#�  G          @�
=?����=q?��Aw33C��?�����
?�p�AY�C��
                                    Bx�i2z  
�          @�Q�?�33��33?uA)�C�b�?�33��(�?L��A��C�U�                                    Bx�iA   T          @��?p������?�Q�A�
=C���?p����ff?��AqG�C���                                    Bx�iO�  �          @�Q�?
=q���R@��A�C�>�?
=q����@
=A�G�C�33                                    Bx�i^l  T          @���?n{���\?��@��C�7
?n{���H>�G�@���C�33                                    Bx�im  T          @�Q�>W
=���\?���AB�\C�,�>W
=���?p��A&ffC�*=                                    Bx�i{�  "          @�  �����
?�=qA>=qC��3������?k�A"ffC��3                                    Bx�i�^  "          @���>u��ff?^�RA�C�P�>u��
=?5@���C�N                                    Bx�i�  �          @���>���\)>�G�@��C��H>����>�z�@H��C��                                     Bx�i��  �          @���?������.{��\)C�B�?����\)���
�`��C�C�                                    Bx�i�P  
�          @���>\��
=��G���G�C��>\���R�
=���C�)                                    Bx�i��  T          @���?+����Ϳc�
�(�C���?+����
����5p�C��                                     Bx�iӜ  T          @�Q�?�G���p����H�V{C�4{?�G���(�����n�\C�AH                                    Bx�i�B  �          @��@	����=q�p���
=C���@	����  �$z����C�)                                    Bx�i��  
�          @�{@Q���G�������
C���@Q��~{�#�
��\C��                                    Bx�i��  T          @�p�@���Ϳ����C���@������
���
C��                                    Bx�j4  "          @�p�?��R��=q�����w
=C���?��R���ÿ��R��=qC�ٚ                                    Bx�j�  T          @�@Q���
=��  ���C��f@Q�����{���C�޸                                    Bx�j+�  
�          @�  @  ��z���
��Q�C���@  ��33�����ffC���                                    Bx�j:&  
�          @�\)@33���Ϳ�����
=C��f@33������H���HC��                                    Bx�jH�  
�          @�ff@�����c�
��RC���@����H��  �2ffC���                                    Bx�jWr  �          @�p�?�(�����p���)G�C�s3?�(���zῆff�<��C��H                                    Bx�jf  �          @�ff?��R��\)���
��\)C��H?��R��{�����p�C�Ф                                    Bx�jt�  "          @�p�?��\��������qC�c�?��\���׿�������C�o\                                    Bx�j�d  "          @��?����
=�˅��C��q?����{��Q����C���                                    Bx�j�
  
�          @�?�p������(��Z=qC���?�p����\�����lz�C��                                    Bx�j��  
�          @�p�@����33����\C��@�����\�#�
��p�C�R                                    Bx�j�V  
�          @�p�?���  ?�\@��RC��)?���Q�>��@�33C��
                                    Bx�j��  
�          @�\)?�����R?��Ahz�C�aH?�����?��HAW
=C�T{                                    Bx�j̢  
�          @�ff@ff���R>��@�Q�C���@ff��
=>\@�
=C���                                    Bx�j�H  �          @�p�@���(�?B�\A�C�N@�����?+�@�RC�E                                    Bx�j��  �          @��@G�����@   A�C���@G���=q?�A��
C��f                                    Bx�j��  
�          @�ff?�G�����@�A؏\C�%?�G����H@�\AУ�C��                                    Bx�k:  T          @�p�?�������@�\A���C�%?�����{@p�A���C��                                    Bx�k�  T          @�ff?��
��{@{A���C�|)?��
��\)@��A�
=C�h�                                    Bx�k$�  �          @�ff?�  ���R@ ��A��
C���?�  ��  ?�
=A�ffC���                                    Bx�k3,  "          @�p�?���
=?�(�A��C�<)?���Q�?��A��
C�+�                                    Bx�kA�  T          @���?Ǯ��ff?�\)Aw\)C�1�?Ǯ��
=?��Ah��C�'�                                    Bx�kPx  �          @�p�?�Q�����?�  A���C��=?�Q���=q?�A�  C�|)                                    Bx�k_  "          @��@%���z�?�  A��\C��{@%����?�
=A�ffC��                                    Bx�km�  �          @��
@*=q���?}p�A2�\C��@*=q��  ?k�A&�\C�H                                    Bx�k|j  T          @�z�@Dz����\>�Q�@�G�C�}q@Dz����H>���@VffC�xR                                    Bx�k�  �          @�z�@C�
����?z�@У�C�� @C�
����?�@�33C���                                    Bx�k��  �          @��@333���\?��HA\(�C�8R@333��33?�33AQG�C�*=                                    Bx�k�\  
�          @��H@P���j=q?�=qAs�C���@P���k�?��
Aj{C���                                    Bx�k�  T          @��
@h���U�?���At��C��=@h���Vff?�ffAlQ�C��
                                    Bx�kŨ  T          @�33@W
=�g�?�G�Ad(�C�q�@W
=�h��?��HA[\)C�aH                                    Bx�k�N  �          @��H@Vff�hQ�?��AP��C�` @Vff�h��?���AH  C�Q�                                    Bx�k��  T          @��@~�R�<��?�AUG�C��R@~�R�>{?��AN�\C���                                    Bx�k�  T          @��@��H�%?h��A$  C��q@��H�&ff?aG�AffC���                                    Bx�l @  �          @�ff@�Q��9��?(��@��C�޸@�Q��:=q?!G�@�{C���                                    Bx�l�  T          @��R@�G��J=q?^�RAG�C���@�G��J�H?Tz�A�HC��3                                    Bx�l�            @�p�@��
�@��?Tz�A�
C��3@��
�@��?L��A{C���                                    Bx�l,2  
�          @�ff@���*=q?:�HA Q�C�p�@���*�H?333@��RC�ff                                    Bx�l:�  T          @��@����?8Q�@�ffC�g�@����?333@�C�^�                                    Bx�lI~  T          @�ff@���z�?(��@�Q�C��f@���z�?!G�@�Q�C���                                    Bx�lX$  �          @�@��\���>��@7
=C�9�@��\���>u@'
=C�5�                                    Bx�lf�  T          @��R@�ff�(Q�>��
@eC���@�ff�(Q�>���@Tz�C��R                                    Bx�lup  �          @�  @�����R?h��A33C���@�����R?aG�A\)C���                                    Bx�l�  T          @���@������?��A?�
C�Ф@����p�?���A<(�C��                                    Bx�l��  T          @�Q�@����
�H?�
=AO\)C�y�@�����?�z�AL(�C�n                                    Bx�l�b  
�          @���@�(���?fffA��C��3@�(�����?c�
AffC���                                    Bx�l�  �          @��@���(�?s33A%�C�e@���p�?p��A"�HC�\)                                    Bx�l��  "          @���@�z����?�  AZ�RC���@�z��{?��RAX��C��=                                    Bx�l�T  �          @�  @�z���R?��Au�C�]q@�z��   ?�\)As\)C�P�                                    Bx�l��  "          @�{@r�\�<��?�33A���C��@r�\�=p�?��A�
=C��q                                    Bx�l�  "          @��@dz��:=q@�
A���C�n@dz��:�H@�\A�
=C�]q                                    Bx�l�F  T          @�z�@^�R�@��@G�A�  C���@^�R�A�@  A�Q�C��H                                    Bx�m�  �          @��\@\(��:=q@AۮC�� @\(��:�H@�A�{C�Ф                                    Bx�m�  "          @��\@L���W�@�A�33C�Ǯ@L���XQ�@G�A��C��)                                    Bx�m%8  
�          @�33@N{�O\)@ ��A�{C�q�@N{�P  @   A��\C�ff                                    Bx�m3�  
Z          @�33@]p��C�
@Q�A�=qC�C�@]p��Dz�@�A��HC�8R                                    Bx�mB�  
�          @��
@W��J=q@p�A�z�C�j=@W��J�H@(�A��C�^�                                    Bx�mQ*  
�          @��@Z=q�U?�\A�
=C��\@Z=q�U?�G�A��
C���                                    Bx�m_�  �          @��H@U�Y��?��HA�C�@ @U�Z=q?ٙ�A���C�9�                                    Bx�mnv  
5          @��
@0�����\?�{Av�HC��@0�����H?���Atz�C��                                    Bx�m}  
(          @���@%��  ?�ffAi�C���@%��  ?��Ag�C���                                    Bx�m��  �          @��@\)���?�=qA=G�C�� @\)���?��A;33C�~�                                    Bx�m�h  �          @�z�@
=����>�(�@�=qC�4{@
=����>�
=@�ffC�4{                                    Bx�m�  T          @��
@�\��p�>��R@aG�C��=@�\��p�>���@Z=qC��=                                    Bx�m��  "          @�?�=q��=q�Ǯ��33C�l�?�=q��녾�����{C�l�                                    Bx�m�Z  �          @�@33���>\)?\C��
@33���>�?�Q�C��
                                    Bx�m�   �          @�{?��
���=�Q�?�  C�?��
���=��
?n{C�                                    Bx�m�  T          @��?����=q=L��?z�C�.?����=q=L��?�C�.                                    Bx�m�L  �          @�?�����=q�8Q��(�C���?�����=q�8Q�� ��C���                                    Bx�n �  
�          @�p�?�G���p�>��R@]p�C��=?�G���p�>��R@\(�C��=                                    Bx�n�  �          @�p�?�\)����?(��@陚C��R?�\)����?(��@陚C��R                                    Bx�n>  �          @�@Q����H?��A<��C�h�@Q����H?��A=�C�j=                                    Bx�n,�  �          @��R@
=��\)?��@�G�C��@
=��\)?��@��C��                                    Bx�n;�  "          @�ff@(����R>�\)@C�
C�}q@(����R>�\)@EC�}q                                    Bx�nJ0  T          @�{@������
�B�\C���@����#�
�\)C���                                    Bx�nX�  
�          @�
=@����þ���
=C���@����ý���{C���                                    Bx�ng|  �          @�
=?�Q����H�B�\�33C��?�Q����H�8Q��p�C��                                    Bx�nv"  �          @�?�G���  >W
=@�C�\?�G���  >aG�@�HC�\                                    Bx�n��  �          @���?Tz���{���
�fffC���?Tz���{��\)�G�C���                                    Bx�n�n  �          @��?����(���=q�>{C�� ?����(�����5C��                                     Bx�n�  �          @�  ?�=q��(�<��
>��C���?�=q��(�=#�
>���C���                                    Bx�n��  T          @��?�����33>8Q�?��RC��?�����33>L��@��C��                                    Bx�n�`  T          @�Q�?�����=q>�=q@;�C�H�?�����=q>�\)@FffC�H�                                    Bx�n�  
�          @���?�ff��=q?(�@���C�1�?�ff���?!G�@ڏ\C�1�                                    Bx�nܬ  
Z          @�Q�?��\��\)>�?�  C�Ф?��\��\)>��?�Q�C�Ф                                    Bx�n�R  4          @�
=?�Q���33?fffA�
C��
?�Q���33?k�A#
=C��R                                    Bx�n��  �          @�Q�?fff���H��G����RC���?fff���H��Q쿂�\C���                                    Bx�o�  T          @��ÿ����\���\�`  C�{�����H��  �\Q�C��                                    Bx�oD  T          @�>�����\)�����K
=C�Q�>�����������G
=C�P�                                    Bx�o%�  �          @��?�  ��
=��{�D��C���?�  ��
=��=q�@��C���                                    Bx�o4�  �          @�Q�@���@G�A��
C�|)@��33@�\A�C���                                    Bx�oC6  �          @���@���\(�@J=qBG�C���@���Z�H@K�B=qC��)                                    Bx�oQ�  T          @���?\�<��@}p�BH�C���?\�;�@~�RBI(�C��q                                    Bx�o`�  
�          @���@ff�Tz�@VffB�C���@ff�S33@W�B�C��
                                    Bx�oo(  T          @�=q@�\�<(�@r�\B633C��{@�\�:=q@s�
B733C�\                                    Bx�o}�  �          @��@�R�P  @c33B'��C�AH@�R�N�R@c�
B)
=C�W
                                    Bx�o�t  �          @���@ff�Q�@e�B*�C�Q�@ff�P��@fffB+�C�g�                                    Bx�o�  �          @��R?��l��@C�
B=qC���?��k�@EBz�C�                                    Bx�o��  �          @��?�p���=q?��RA��C�S3?�p�����@ ��A�(�C�\)                                    Bx�o�f  �          @�(�@
�H��  ?�A��\C���@
�H���?�\)A�33C���                                    Bx�o�  �          @���@���G�@G�A�ffC��R@�����@33A��C���                                    Bx�oղ  �          @��@Q��0��@h��B3�
C�l�@Q��.�R@j=qB5
=C��                                    Bx�o�X  �          @���@���S33@N{B�C�7
@���Q�@O\)Bp�C�N                                    Bx�o��  �          @�z�@Q��s33@.�RB �\C��H@Q��q�@0��B  C���                                    Bx�p�  �          @��
?�p��r�\@4z�B�HC���?�p��p��@6ffB\)C��q                                    Bx�pJ  �          @��
?����xQ�@,��A�
=C�Q�?����w
=@.�RB{C�aH                                    Bx�p�  �          @��
?�=q�S�
@Z�HB(�C�p�?�=q�R�\@\��B*
=C��=                                    Bx�p-�  
�          @��
@ ���xQ�@(Q�A���C��R@ ���w
=@*�HA��
C���                                    Bx�p<<  �          @��
?ٙ���p�@ffA��HC�
?ٙ�����@��A�Q�C�"�                                    Bx�pJ�  �          @��?�\)����?�A��\C���?�\)����?��HA�  C���                                    Bx�pY�  �          @�  ?У��k��U��C���?У��l���S�
�(�C��                                    Bx�ph.  �          @�  ?����o\)�G
=�(�C��?����qG��E��ffC��)                                    Bx�pv�  �          @�{@ff��z���
��(�C�p�@ff������У�C�b�                                    Bx�p�z  �          @��R@��}p�����ڏ\C�o\@��\)�ff��
=C�^�                                    Bx�p�   �          @�{@E��c33�
=q��G�C�y�@E��dz��Q���(�C�ff                                    Bx�p��  �          @��
@*=q��������\)C�޸@*=q������
=����C��)                                    Bx�p�l  �          @�33@z���ff�:�H�G�C�@z����R�0�����C���                                    Bx�p�  �          @��@
=q��G����
�8��C��3@
=q�����}p��1G�C���                                    Bx�pθ  T          @��?�
=���\�����x��C�l�?�
=���H����pz�C�e                                    Bx�p�^  �          @�z�?�33��������@z�C�R?�33��p����
�8Q�C�3                                    Bx�p�  �          @�@	�����Ϳ0������C�j=@	�����Ϳ#�
����C�ff                                    Bx�p��  �          @�ff@1G����
��ff��C�(�@1G���(�������ffC�&f                                    Bx�q	P  �          @�(�@(Q�����?��@�Q�C��R@(Q�����?��@�  C��q                                    Bx�q�  �          @�z�@/\)����>��@�Q�C�Z�@/\)����?�\@��C�^�                                    Bx�q&�  �          @��@>�R����>8Q�@z�C���@>�R����>k�@#33C��
                                    Bx�q5B  �          @��
@0�����?Q�A��C��@0����33?\(�Ap�C��3                                    Bx�qC�  �          @�(�@AG��[�?��RA��
C��
@AG��Z=q@G�A�p�C��                                    Bx�qR�  �          @�=q@B�\�mp�?��Ap��C��@B�\�l(�?�=qAxQ�C���                                    Bx�qa4  �          @��H@qG��AG�?˅A��C���@qG��@  ?�\)A���C��q                                    Bx�qo�  �          @�p�@�{�8��?Y��A  C���@�{�8Q�?aG�A��C��H                                    Bx�q~�  T          @�z�@�ff�:�H>�
=@�C��R@�ff�:�H>�ff@���C��q                                    Bx�q�&  �          @�33@u�AG�?�=qAt��C��@u�@  ?�\)Az�HC��)                                    Bx�q��  �          @�Q�@u��0  ?�
=A���C�&f@u��.�R?��HA��C�=q                                    Bx�q�r  �          @���@Y���Mp�?�\)A��\C�J=@Y���L(�?�z�A�  C�b�                                    Bx�q�  �          @�G�@I���P  @�RAљ�C��@I���N{@��A�G�C�,�                                    Bx�qǾ  �          @��H@;��l��?��A��C�33@;��k�?�A��C�G�                                    Bx�q�d  �          @��@=p�����?��ADQ�C�.@=p���Q�?���AL��C�8R                                    Bx�q�
  �          @��@A��g
=?޸RA�{C�  @A��e?��
A�{C�3                                    Bx�q�  �          @��H@E��I��@"�\A�p�C�4{@E��G�@$z�A��C�W
                                    Bx�rV  �          @�G�@5�n�R?�=qA�z�C���@5�mp�?�\)A���C��R                                    Bx�r�  �          @�  ?�(����R?��\A=G�C��=?�(���ff?���AF�HC���                                    Bx�r�  �          @�Q�?��
����?�z�A��
C�P�?��
��Q�?�(�A��HC�Y�                                    Bx�r.H  �          @��?����{?���Aq�C�H?����p�?�\)A|(�C��                                    Bx�r<�  �          @���?�  ��G�?�G�Ag�C���?�  ����?��Aq��C��3                                    Bx�rK�  �          @��@�R����?��HA�
=C��
@�R��  ?�  A��C��                                    Bx�rZ:  �          @�G�@{���H?�  Ae�C��f@{���\?�ffAn�RC���                                    Bx�rh�  �          @���@����G�?�{Ay�C��q@������?�z�A�C���                                    Bx�rw�  �          @�Q�@Q���(�?�p�A�(�C��
@Q����
?��
A��HC�                                    Bx�r�,  �          @�z�@
�H���?Y��Ap�C��
@
�H����?h��A#33C��q                                    Bx�r��  �          @���@
=q��z�>�@��
C�}q@
=q��z�?
=q@��C��H                                    Bx�r�x  �          @���@
=���?\)@�\)C�0�@
=����?(�@ۅC�4{                                    Bx�r�  �          @�(�@���Q�>Ǯ@�p�C��)@���  >�ff@���C��                                     Bx�r��  �          @��@z����R?�z�AX��C�s3@z���ff?��HAb�\C�}q                                    Bx�r�j  �          @��\@33�z=q@ ��A�
=C��3@33�w�@#33A��
C��=                                    Bx�r�  �          @�=q@�
�}p�@
=qA�G�C��@�
�|(�@��A�  C�0�                                    Bx�r�  �          @�(�@{����?�p�A��
C�\)@{��(�?��
A���C�l�                                    Bx�r�\  �          @�33@ ����\)?���AuG�C�]q@ �����R?�33A~�RC�h�                                    Bx�s
  �          @��
@   ��33?�  A�Q�C���@   ��=q?�ffA�
=C��H                                    Bx�s�  �          @��@�
����@�A�z�C�� @�
����@�RA�33C��{                                    Bx�s'N  �          @���@ff���
?�p�A�p�C�ٚ@ff���H@�A�=qC��                                    Bx�s5�  �          @���@��G�@Q�A�  C�\@��Q�@�AȸRC�#�                                    Bx�sD�  �          @��@����@�A�(�C�
@��  @{A���C�+�                                    Bx�sS@  �          @��@��33?��
A�z�C�Ф@��=q?�A�p�C�޸                                    Bx�sa�  �          @���?�  ���R?�\)A���C��3?�  ��{?�
=A�{C��                                     Bx�sp�  �          @��?���{?�{A�{C�)?����?�A�33C�(�                                    Bx�s2  �          @�p�?�33���?�(�A��C��?�33���R@�A���C�"�                                    Bx�s��  
�          @�?Ǯ����?��HA�Q�C�|)?Ǯ��  @G�A��C���                                    Bx�s�~  �          @�p�?�=q���
@�\A���C�W
?�=q��33@ffA��
C�e                                    Bx�s�$  T          @�p�?�{��=q@ ��A�
=C���?�{����@�
A�{C��
                                    Bx�s��  T          @�p�?����{?�A��HC�z�?����p�?��A��
C���                                    Bx�s�p  �          @�{?�����p�?�ffA�z�C�8R?�������?�{A��C�AH                                    Bx�s�  �          @���?���33?�=qA�ffC���?����\?��A��C���                                    Bx�s�  �          @�{@���Q�@�
A�p�C�4{@����@
=A�=qC�E                                    Bx�s�b  �          @�@(���ff@ffA�33C��f@(���p�@	��A��C��R                                    Bx�t  �          @�z�@���G�?�z�A���C��3@�����?��HA���C��                                    Bx�t�  �          @���@���=q?�\)A���C��R@�����?�A��C��                                    Bx�t T  �          @��@#33�\)@�A�  C�O\@#33�}p�@�A�z�C�b�                                    Bx�t.�  �          @�@!���33?��A���C���@!����\?�
=A�\)C���                                    Bx�t=�  �          @��R@#�
��33?��HA��RC��)@#�
��=q@ ��A��C�\                                    Bx�tLF  �          @���@\)��=q?�
=A�(�C��
@\)����?�p�A��\C���                                    Bx�tZ�  �          @��
@Q����\@\)Aϙ�C�Ǯ@Q�����@�A�(�C���                                    Bx�ti�  T          @�33?������@�RA�G�C�?�����G�@!�A��C�!H                                    Bx�tx8  T          @�(�@p���Q�?޸RA�p�C���@p����?��A��C�                                    Bx�t��  �          @�z�@(����?�{A��
C�Q�@(����H?�33A�(�C�b�                                    Bx�t��  �          @���@"�\���?�33A��C���@"�\��z�?ٙ�A�(�C��                                     Bx�t�*  �          @�(�@$z���(�?�\)A��C���@$z����?�A��C��                                    Bx�t��  �          @�{@1G����?�A��\C�+�@1G���G�?�(�A���C�:�                                    Bx�t�v  �          @��@4z����\?�A���C�L�@4z���=q?��HA���C�\)                                    Bx�t�  T          @��R@*=q���?��
A�C�o\@*=q���H?���A�C�~�                                    Bx�t��  �          @��R@!����R?޸RA�C�z�@!���ff?��
A��
C���                                    Bx�t�h  �          @�{@#�
��ff?�
=A�G�C��@#�
��?�p�A�G�C��)                                    Bx�t�  T          @�  @*=q���H?�Q�A��C���@*=q��=q?�p�A���C��{                                    Bx�u
�  �          @��R@,(��y��@	��A�
=C�O\@,(��xQ�@(�A���C�b�                                    Bx�uZ  �          @�@.{�s�
@
�HA�\)C�� @.{�r�\@p�A��C��3                                    Bx�u(   T          @�ff@?\)�i��@	��A�ffC���@?\)�g�@(�A��
C�                                    Bx�u6�  �          @�@B�\�n{?��A�\)C���@B�\�l��?�A���C��3                                    Bx�uEL  �          @�ff@A��n�R?�Q�A��C��f@A��mp�?�(�A��C��R                                    Bx�uS�  �          @��R@���{?�33A�G�C�Ff@���?�Q�A�
=C�S3                                    Bx�ub�  �          @�\)@AG��_\)@�HA�
=C�q�@AG��^{@��A�=qC��=                                    Bx�uq>  �          @�ff@?\)�p��?���A���C�Ff@?\)�o\)?�p�A��C�W
                                    Bx�u�  �          @���@*�H����?޸RA�ffC�˅@*�H��Q�?��
A��
C�ٚ                                    Bx�u��  �          @��@,���xQ�?�33A�p�C�o\@,���w
=?�Q�A���C�}q                                    Bx�u�0  �          @�Q�@{�]p�@(��A���C���@{�\(�@*�HB
=C���                                    Bx�u��  �          @�\)@#33�Z�H@#�
A�33C�]q@#33�Y��@%A�Q�C�s3                                    Bx�u�|  �          @��R@3�
�Vff@A�RC��@3�
�U�@
=A�C�R                                    Bx�u�"  �          @�G�@@  �]p�@�\A�  C�w
@@  �\��@z�A��HC���                                    Bx�u��  T          @�33@E��\(�@
=A�z�C��\@E��Z�H@��A�G�C��                                    Bx�u�n  �          @�  @`���<(�?&ffA��C�
=@`���;�?.{A=qC��                                    Bx�u�  �          @�=q@k��?\)�@  ��C�s3@k��@  �:�H�
=C�k�                                    Bx�v�  �          @�=q@k��C�
�u�5C��@k��C�
�W
=�#�
C�)                                    Bx�v`  �          @��
@P���a�>�Q�@��C�^�@P���a�>\@���C�aH                                    Bx�v!  T          @�Q�@7��^�R?�(�A��HC���@7��^{?�  A�p�C��{                                    Bx�v/�  �          @��R@W��\(�?���AS�C�33@W��[�?�z�AXQ�C�:�                                    Bx�v>R  �          @��
@k��b�\=���?��C��@k��b�\=�?�\)C�3                                    Bx�vL�  �          @��\@k��[�>�G�@��C�� @k��[�>�@�  C���                                    Bx�v[�  �          @��@^{�W��W
=�p�C���@^{�W��B�\���C��                                    Bx�vjD  �          @�
=@HQ��]p���p���z�C��@HQ��^{���H��Q�C�                                    Bx�vx�  �          @�Q�@<(��S33�ٙ����C��R@<(��S�
��
=���
C���                                    Bx�v��  T          @���@Mp����\@R�\B.Q�C�<)@Mp���  @S33B.��C�b�                                    Bx�v�6  �          @�=q@`�׿k�@G�B"�C���@`�׿fff@G�B"�
C��{                                    Bx�v��  �          @�
=@hQ��\)����G�C���@hQ��\)���Ϳ�=qC�Ǯ                                    Bx�v��  T          @��@"�\��{@XQ�BAp�C���@"�\�˅@X��BB{C��
                                    Bx�v�(  �          @��\@Q��:=q?�\)Af{C�0�@Q��:=q?��AiG�C�7
                                    Bx�v��  T          @���@aG��>{?��Az�RC��@aG��=p�?��A}��C��{                                    Bx�v�t  T          @��@XQ��7�?��A�p�C��@XQ��7
=?��A���C�ٚ                                    Bx�v�  T          @��@^{�8��?��A���C�)@^{�8Q�?��A��C�&f                                    Bx�v��  T          @�(�@c�
�>{?��A��RC��@c�
�>{?�ffA�  C�{                                    Bx�wf  �          @�z�@[��5?�p�A��C�7
@[��5�?��RA���C�AH                                    Bx�w  �          @��\@fff�3�
?��
A�C�H@fff�3�
?��A��HC��                                    Bx�w(�  �          @��@u��.�R?�Q�Ab�RC�AH@u��.�R?���Ad��C�Ff                                    Bx�w7X  �          @��@n{�-p�?��
Ay�C��q@n{�,��?��Az�HC��                                    Bx�wE�  
�          @�\)@W
=�P  ?�\A��
C��@W
=�O\)?��
A��HC��                                    Bx�wT�  �          @��@I���N{@
=qA�z�C�/\@I���N{@
�HA�p�C�5�                                    Bx�wcJ  �          @��@\���5@�A�G�C�B�@\���5�@�A�  C�J=                                    Bx�wq�  �          @���@U��J=q@ffA�Q�C�@ @U��J=q@
=A�
=C�Ff                                    Bx�w��  �          @�ff@I���-p�@%A�G�C�� @I���-p�@%A��C���                                    Bx�w�<  �          @�@Fff���@:�HB�RC�%@Fff���@:�HB��C�,�                                    Bx�w��  �          @�{@`���   @333B
�
C�'�@`���   @333B
=C�/\                                    Bx�w��  T          @��R@`  �!�@p�A���C�3@`  �!�@p�A�\)C�R                                    Bx�w�.  �          @��@S33�7�@
=A���C��H@S33�7
=@
=A�G�C���                                    Bx�w��  �          @���@<(��<(�@7
=B
(�C�y�@<(��<(�@7
=B
G�C�|)                                    Bx�w�z  �          @��@G��5�@2�\B��C��\@G��5�@2�\B
=C��                                    Bx�w�   �          @�Q�@G��Q�@y��BM
=C�(�@G��Q�@y��BM{C�*=                                    Bx�w��  T          @���@
=��R@j=qB;�RC��H@
=��R@j=qB;�RC��                                     Bx�xl  �          @�  @ff� ��@hQ�B:(�C��{@ff� ��@hQ�B:{C���                                    Bx�x  �          @�
=@	����H@p  BD�
C���@	����H@p  BD�RC���                                    Bx�x!�  �          @���?���%�@w�BK�
C�Y�?���%@w�BK��C�T{                                    Bx�x0^  �          @�=q@"�\�p�@hQ�B6�C��@"�\�{@g�B6�RC��                                    Bx�x?  �          @�=q@&ff�=q@c33B4Q�C���@&ff�=q@c33B4{C���                                    Bx�xM�  �          @�(�@�� ��@{�BIQ�C��{@�� ��@z�HBH��C��=                                    Bx�x\P  �          @���@(Q��1G�@\(�B'�HC��@(Q��1�@[�B'�C���                                    Bx�xj�  �          @��
@'
=�=p�@P��B33C���@'
=�>{@P  BC��H                                    Bx�xy�  �          @��@!��I��@G
=B=qC�^�@!��J=q@FffBC�U�                                    Bx�x�B  �          @�z�@
=� ��@qG�B>z�C��R@
=�!G�@p��B=��C���                                    Bx�x��  �          @���@"�\�R�\@C33B�C�ٚ@"�\�S33@B�\BQ�C��\                                    Bx�x��  �          @��
@!��R�\@:�HB�C���@!��S33@:=qB�
C��H                                    Bx�x�4  �          @�z�@(Q��O\)@>�RB�C��@(Q��P��@>{BffC�xR                                    Bx�x��  
�          @��
@ff�Z�H@>{B(�C�E@ff�[�@<��BQ�C�8R                                    Bx�xр  �          @�G�@3�
�8��@8Q�B\)C��@3�
�9��@7�B��C��                                    Bx�x�&  �          @�
=@<(��G�@S�
B&��C�,�@<(���\@S33B&�C�3                                    Bx�x��  �          @�\)@(���R@c33B6ffC�G�@(��\)@a�B5�C�+�                                    Bx�x�r  �          @�z�@333��@K�B*�C�f@333�33@J�HB)�RC��                                    Bx�y  �          @�(�@qG�?�
=@(�A߮A�  @qG�?�z�@��A���A��R                                    Bx�y�  �          @�Q�@5���33@-p�B.
=C�o\@5���p�@-p�B-�
C�C�                                    Bx�y)d  �          @�Q�@���^{@ffAҸRC���@���_\)@�A�(�C���                                    Bx�y8
  �          @��
@XQ�Ǯ@@  B=qC��
@XQ��=q@>�RB�C�p�                                    Bx�yF�  �          @��@6ff��Q�@o\)BJ
=C���@6ff��(�@n�RBI\)C�l�                                    Bx�yUV  �          @�33@P��=#�
@`  B<�?E�@P��<��
@`  B<(�>�z�                                    Bx�yc�  �          @�G�@J=q?
=q@`��B>G�Aff@J=q?�\@aG�B>�\A=q                                    Bx�yr�  �          @���@\�Ϳ:�H@S�
B,��C�
=@\�ͿB�\@S�
B,\)C���                                    Bx�y�H  �          @��
@X�ÿ���@K�B&33C�<)@X�ÿ�p�@J�HB%�C�                                    Bx�y��  �          @��
@�=q���@33A�\)C���@�=q���@�\A�Q�C���                                    Bx�y��  �          @�@c33���@J=qB!z�C�t{@c33��\)@I��B �
C�:�                                    Bx�y�:  �          @���@N�R�\@N{B(
=C�n@N�R��ff@L��B'
=C�33                                    Bx�y��  �          @�(�@R�\�У�@K�B#p�C�˅@R�\��@J=qB"\)C���                                    Bx�yʆ  �          @���@J=q���
@FffB!��C�C�@J=q����@E�B �RC�
=                                    Bx�y�,  �          @��@)���   @G�B"G�C�U�@)���"�\@EB �C�"�                                    Bx�y��  �          @�{@mp�����@  A��HC�U�@mp���\)@�RA���C�*=                                    Bx�y�x  �          @�=q@�ff?5?5A�A
=@�ff?333?8Q�A\)Ap�                                    Bx�z  T          @�Q�@��H?��R>#�
?�ffAf=q@��H?��R>.{?�(�Ae�                                    Bx�z�  �          @��\@���>�(�?�
=A���@�(�@���>��?�Q�A�33@�(�                                    Bx�z"j  �          @�p�@�?aG�?�  A`��A!G�@�?\(�?�G�Ab�HAff                                    Bx�z1  �          @��@�G�?Tz�?��AI�AG�@�G�?O\)?�33AK
=A�\                                    Bx�z?�  �          @�ff@�ff?��?�
=A�z�@ۅ@�ff?�?�Q�A�33@�(�                                    Bx�zN\  �          @�z�@��
?+�?�  A�(�@�G�@��
?#�
?�G�A�
=@�G�                                    Bx�z]  �          @�@�z�?
=?�{A�
=@��@�z�?�?�\)A��
@�z�                                    Bx�zk�  �          @�@��?(�?ǮA�{@�=q@��?z�?���A��H@�G�                                    Bx�zzN  �          @��@��>�G�?�p�A]G�@��@��>�
=?��RA^ff@���                                    Bx�z��  �          @�z�@��
?0��?�Q�A�=qA   @��
?(��?���A�G�@��                                    Bx�z��  �          @��@��?��R?��HA��RAk33@��?�(�?�p�A��\Af�R                                    Bx�z�@  �          @��@����33@p�A�z�C�� @����Q�@�A�  C��                                     Bx�z��  �          @�p�@��׿�Q�@$z�AC�'�@��׿�p�@"�\A�(�C��f                                    Bx�zÌ  �          @�{@k��p�@+�A�C�%@k��   @(��A�33C���                                    Bx�z�2  
�          @��R@qG��
=q@7
=B=qC��@qG��p�@4z�B(�C��3                                    Bx�z��  T          @�\)@u����@(��A�z�C�@u��(�@%A��C�                                    Bx�z�~  �          @�
=@tz���H@%�A�G�C��@tz��{@!�A�\C���                                    Bx�z�$  �          @�@^{�1G�@*=qA�G�C��3@^{�4z�@&ffA�C�q�                                    Bx�{�  �          @�
=@Vff�7
=@1�B�C���@Vff�:�H@.�RA�=qC�u�                                    Bx�{p  �          @�33@W
=�2�\@(Q�A��\C�  @W
=�6ff@$z�A�z�C���                                    Bx�{*  �          @���@Y���.�R@#33A��C���@Y���1�@   A�
=C�Y�                                    Bx�{8�  �          @�  @X���+�@\)A��C��3@X���/\)@�A���C��                                    Bx�{Gb  �          @�p�@dz��   @G�A�=qC�}q@dz��#33@{A�ffC�:�                                    Bx�{V  �          @�@-p��0  @7
=B(�C�J=@-p��4z�@333Bz�C��
                                    Bx�{d�  �          @�@�{@k�BC(�C�<)@�33@hQ�B?z�C��{                                    Bx�{sT  �          @��@��g�@�HA�{C���@��k�@A�RC�xR                                    Bx�{��  �          @�G�?��fff@�A��HC�U�?��i��@  A��C�(�                                    Bx�{��  �          @��H@�z���?uA9�C��H@�z��?s33A7\)C�}q                                    Bx�{�F  �          @�\)@�p��z�>���@�z�C��@�p��
=>Ǯ@�
=C���                                    Bx�{��  �          @�  @��R��G�>k�@&ffC�t{@��R��ff>aG�@{C�k�                                    Bx�{��  �          @�=q@���>��#�
��ff@�  @���>������R@�Q�                                    Bx�{�8  �          @���@�?s33�aG��"�\A.�R@�?u�L���  A/�                                    Bx�{��  �          @�  @�?W
=�#�
��{A�H@�?W
=�\)����A�                                    Bx�{�  �          @�\)@�z�?G�=�\)?W
=AQ�@�z�?E�=�Q�?�=qA�
                                    Bx�{�*  �          @�  @�
=>�=q>\@�33@E�@�
=>��>Ǯ@�@=p�                                    Bx�|�  �          @��R@��?�>�\)@N{@�=q@��?\)>�z�@Y��@�\)                                    Bx�|v  �          @�ff@���?�\>��H@�z�@�\)@���?   ?   @��@�=q                                    Bx�|#  �          @�p�@��
?z�>���@w
=@أ�@��
?�>�33@���@��                                    Bx�|1�  �          @�33@��>u>��@��@5@��>aG�>�@���@*�H                                    Bx�|@h  �          @��H@���>B�\?�@�  @\)@���>.{?�@��@�                                    Bx�|O  �          @�=q@�p���z�?��AR{C�:�@�p����
?�=qAP(�C�f                                    Bx�|]�  �          @�Q�@��#�
?L��AffC�@��B�\?J=qAp�C��q                                    Bx�|lZ  �          @��\@�=q���?�ffA?�
C��@�=q����?�G�A9p�C���                                    Bx�|{   �          @��@��ͿǮ?�
=A��C���@��Ϳ�{?�\)AyC�y�                                    Bx�|��  �          @��H@�{�Ǯ?��HA\z�C�Ф@�{����?�z�AR�HC���                                    Bx�|�L  �          @�z�@�����?Tz�A��C��H@������?J=qA��C�z�                                    Bx�|��  �          @�p�@�33����?���AEp�C�\@�33��?��A<��C��)                                    Bx�|��  �          @�{@�G����?���A>�\C���@�G���
=?�G�A4(�C�W
                                    Bx�|�>  �          @�ff@�p���G�?�@�=qC�|)@�p����
?�@�\)C�b�                                    Bx�|��  �          @�ff@�G���z�?333@�ffC�"�@�G���
=?!G�@�{C��                                    Bx�|�  �          @��R@�z����?uA*{C��R@�z���?aG�A(�C�l�                                    Bx�|�0  �          @�\)@�  ��?�33Ax  C���@�  ��?��Ah��C�P�                                    Bx�|��  
�          @�Q�@����   @G�A�\)C��@�����?�Q�A�(�C�^�                                    Bx�}|  �          @�  @���33?�p�A���C��@���
=?�33Ax��C���                                    Bx�}"  �          @���@�=q�33?��A��\C���@�=q�
=?ǮA�G�C��)                                    Bx�}*�  �          @���@������?�G�A�ffC�}q@���\?ٙ�A�
=C��                                    Bx�}9n  �          @���@�p���
=?��@�C��@�p�����?
=q@���C��f                                    Bx�}H  �          @���@�{��녿��
�h(�C��H@�{��Q쿦ff�j�HC��                                    Bx�}V�  z          @�33@�\)����33��33C��=@�\)=#�
��33��33?�                                    Bx�}e`  �          @�33@��׾�p��޸R���C��=@��׾�����  ��
=C�8R                                    Bx�}t  �          @�  @��\�W
=��
=����C��
@��\����Q���\)C�8R                                    Bx�}��  �          @�=q@��\���
���R�g�C��q@��\���\��Q����
C��                                    Bx�}�R  �          @���@��ÿ�=q@A��\C�4{@��ÿ�@�A��C��R                                    Bx�}��  T          @��@�ff��@z�A�G�C��R@�ff��G�@ ��A�\)C��                                    Bx�}��  
�          @��@�z�\)?�{AAC��@�z�(�?�=qA=�C���                                    Bx�}�D  �          @���@��H�B�\?�  A��C���@��H�Q�?��HA|��C�g�                                    Bx�}��  �          @�\)@�33�J=q?=p�@��\C�Ǯ@�33�Tz�?5@�{C���                                    Bx�}ڐ  �          @�p�@��ÿ��>�p�@z=qC�,�@��ÿ�{>��
@XQ�C�
                                    Bx�}�6  �          @��H@�����(�?E�A��C���@�����  ?333@���C��
                                    Bx�}��  �          @�
=@����ff?�=qA`��C�R@����{?��RAQC��=                                    Bx�~�  �          @�
=@�(����R?�Q�At  C���@�(��33?��Ac\)C��
                                    Bx�~(  �          @�
=@�\)���?���A:�HC���@�\)��
=?�G�A+
=C�ff                                    Bx�~#�  �          @���@��\��\)?��A-G�C���@��\��?p��A��C��f                                    Bx�~2t  
�          @�(�@�  ��p�>Ǯ@|(�C��=@�  �   >�z�@<(�C���                                    Bx�~A  �          @��\@�(��(��<#�
>�C��=@�(��(�ý���p�C��                                    Bx�~O�  �          @��@���%��s33�(�C�H@���!녿�=q�1�C�=q                                    Bx�~^f  
�          @��
@���녿���<z�C��
@���{��G��O�
C��                                    Bx�~m  
�          @��@����/\)�(���33C�'�@����,�ͿB�\��=qC�Q�                                    Bx�~{�  �          @���@��׿�Q���R��
=C�` @��׿�������Q�C�f                                    Bx�~�X  �          @��H@~�R��z��I���C��\@~�R���R�O\)�=qC���                                    Bx�~��  �          @�p�@�  ��
=�L���p�C��
@�  ��G��Q��
=C��=                                    Bx�~��  �          @�z�@p���
�H�K���C��@p�׿��R�R�\��RC�
=                                    Bx�~�J  �          @�z�@{���\�4z���ffC��f@{��Q��<(��\)C�Ǯ                                    Bx�~��  T          @�@����%������C�h�@����p����Ə\C�f                                    Bx�~Ӗ  
�          @�33@�z���Ϳfff�33C��@�z���ÿ��
�333C�]q                                    Bx�~�<  �          @��
@��H�%��Q����C���@��H�\)��=q���HC�R                                    Bx�~��  �          @�(�@�  �-p���ff���C��@�  �&ff�������C�J=                                    Bx�~��  �          @�G�@y������(Q���C�:�@y����R�1G���33C��                                    Bx�.  �          @�@�  �(Q쿪=q�d  C�ٚ@�  �#33��p��
=C�>�                                    Bx��  T          @��@�{�8�ÿ�\)��C��@�{�1���\��
=C�9�                                    Bx�+z  T          @���@z=q�E���R�_\)C��q@z=q�@�׿�����C�:�                                    Bx�:   T          @�33@�{�*�H�#�
���
C�}q@�{�*=q�.{��\)C���                                    Bx�H�  �          @�
=@�z��<�;����G�C�
=@�z��:�H�&ff��  C�1�                                    Bx�Wl  �          @�
=@��\�O\)�����HC��)@��\�L�Ϳ:�H��
=C��f                                    Bx�f  �          @��R@���X�ÿ�\��(�C��\@���Vff�8Q����
C��R                                    Bx�t�  �          @�\)@�Q��j=q�5��C��=@�Q��g
=�p���{C���                                    Bx��^  �          @�Q�@Z=q�~{�ٙ���{C�O\@Z=q�w
=������Q�C��)                                    Bx��  �          @���@g��k���z���z�C�Ff@g��c33�	�����C�˅                                    Bx���  �          @�
=@�Q��	��>Ǯ@�(�C�]q@�Q���>�  @(��C�C�                                    Bx��P  �          @�{@��H�(�>��@�p�C���@��H�p�>���@Mp�C��                                    Bx���  �          @���@�����>��R@UC�Ǯ@���=q>��?У�C��{                                    Bx�̜  �          @�(�@�\)�!G�>���@c�
C��R@�\)�"�\>#�
?�p�C��                                    Bx��B  
�          @��
@����-p�=�?�G�C�~�@����.{��\)�B�\C�}q                                    Bx���  T          @�p�@�p��.�R����˅C��)@�p��-p���{�e�C��\                                    Bx���  �          @�{@����<(��z��ÅC�|)@����8�ÿG��Q�C���                                    Bx��4  �          @�  @����J=q�����b�\C�޸@����C33�Ǯ��(�C�P�                                    Bx���  �          @���@�{�P�׿�  �|��C��@�{�H�ÿ�p���{C��{                                    Bx��$�  �          @���@���L�Ϳ�����C�
=@���Dz�������C��                                    Bx��3&  �          @��@����E�p���p�C�W
@����:�H��H��33C��                                    Bx��A�  �          @�ff@l(��Fff�#�
�߮C�H@l(��9���1�����C��                                    Bx��Pr  �          @�ff@k��9���2�\��{C���@k��,(��?\)�=qC���                                    Bx��_  T          @�\)@\)�g
=�h���{C��{@\)�a녿�
=�G�C�C�                                    Bx��m�  �          @���@tz��S�
�Q�����C���@tz��H�������HC�H�                                    Bx��|d  �          @�Q�@h���K��(����p�C�k�@h���>{�7����
C�aH                                    Bx���
  �          @�Q�@a��@���;���C��@a��1G��I���=qC��\                                    Bx����  T          @��@{��0  �2�\��C�|)@{��!��?\)���C���                                    Bx���V  �          @��
@z�H�Fff�'
=��{C��R@z�H�8���5��C���                                    Bx����  �          @��\@����<(�>aG�@��C���@����<�ͼ#�
���C���                                    Bx��Ţ  T          @�33@�\)�Dz῁G��(��C��=@�\)�>�R��  �Q��C�/\                                    Bx���H  �          @�=q@���A�?&ff@أ�C�.@���E�>��@�
=C���                                    Bx����  �          @���@�p��S�
>���@[�C��)@�p��Tz�=u?&ffC��=                                    Bx���  �          @��H@���S33>��
@O\)C��
@���S�
=L��>��C�Ǯ                                    Bx�� :  �          @�p�@�
=�\(�=#�
>��C�0�@�
=�\(��u�p�C�9�                                    Bx���  �          @�p�@�=q�l��?Tz�A	�C�޸@�=q�p  ?�@��
C���                                    Bx���  �          @��R@��R�e�?���AC�C�˅@��R�j�H?c�
Ap�C�s3                                    Bx��,,  �          @�{@�  �c33?���AD(�C�\@�  �h��?fffA=qC��
                                    Bx��:�  T          @�z�@G
=�q�@/\)A�C��{@G
=��  @��A�
=C��                                    Bx��Ix  �          @�=q@L(�����?\A}�C�B�@L(���p�?��A=��C��f                                    Bx��X  �          @�33@y���j�H?�z�AD  C�aH@y���p��?W
=AG�C��                                    Bx��f�  �          @�(�@��H�L��@ffA��\C��R@��H�W�?���A��C�>�                                    Bx��uj  �          @�G�@��>�R?���A��\C�B�@��H��?�
=A��RC���                                    Bx���  �          @��@s�
�>�R@7�A�\)C���@s�
�N{@%AۮC��H                                    Bx����  �          @�ff@��
�Fff?��A���C�K�@��
�P��?���A���C��H                                    Bx���\  �          @���@���;�?���AD��C��=@���A�?n{A  C��                                     Bx���  �          @�(�@��
�1G�>�Q�@j�HC�0�@��
�2�\=�G�?��C��                                    Bx����  �          @���@�{��@�
A�=qC��{@�{�$z�@A���C��q                                    Bx���N  
�          @�G�@�=q�%�?c�
A
=C�~�@�=q�*=q?&ff@�Q�C�.                                    Bx����  T          @�Q�@����R?��AS\)C��f@���%?���A,��C�E                                    Bx���  T          @�@��
�  ?��
A&ffC�&f@��
��?O\)A\)C���                                    Bx���@  "          @�\)@��Ϳ��H?�p�A��C�ٚ@��Ϳ�{?ǮA~�RC�3                                    Bx���  T          @���@�
=�,(�?�A:�\C���@�
=�2�\?fffAQ�C�T{                                    Bx���  
�          @���@��ff@
=A\C���@��$z�@Q�A�  C��                                     Bx��%2  �          @���@��ÿ��@\(�B��C��H@��ÿ�(�@QG�B  C���                                    Bx��3�  T          @�\)@*�H=�\)@��Bt��?�@*�H���@��HBsffC���                                    Bx��B~  "          @��@J=q�:�H@��RB^
=C��H@J=q��Q�@��
BW�C���                                    Bx��Q$  "          @��@�ff�W
=@tz�B&�C�S3@�ff��(�@mp�B!z�C��                                    Bx��_�  
�          @��@�=q�}p�@���B/��C�&f@�=q���@z�HB)\)C���                                    Bx��np  �          @��@��ÿk�@�
=B6p�C��@��ÿ��@��B033C��=                                    Bx��}  �          @�z�@�G��\)@~{B*  C�B�@�G��u@x��B%�C��\                                    Bx����  T          @��
@�
=��G�@|(�B)  C�G�@�
=��33@tz�B"��C�˅                                    Bx���b  �          @��
@vff�z�H@��B==qC���@vff��@�{B6=qC���                                    Bx���  �          @�33@����O\)@uB$�HC���@�������@n�RBz�C�1�                                    Bx����  	�          @���@�\)�^�R@c�
B�HC���@�\)���R@\��Bp�C�J=                                    Bx���T  
�          @��H@����Q�@aG�B��C���@����ff@XQ�B��C�w
                                    Bx����  
�          @��@�  �}p�@eB�HC��@�  ��{@^{B�RC���                                    Bx���  
�          @�(�@�ff�0��@�G�B-��C�O\@�ff��\)@|(�B(��C���                                    Bx���F  
�          @���@�33��G�@\(�B�\C��{@�33��\)@S�
B
\)C���                                    Bx�� �  
�          @��R@���!G�@��\B.Q�C��@�����@\)B)z�C��
                                    Bx���  T          @�G�@�\)�5@U�B�C���@�\)����@N�RB�\C��
                                    Bx��8  �          @��@��^�R@Y��B�C�H@����R@Q�BG�C��
                                    Bx��,�  T          @\@|(���G�@��BDp�C�<)@|(��(�@���BB(�C���                                    Bx��;�  
�          @��
@�\)�Ǯ@�(�B*G�C�}q@�\)�Y��@��B&C��                                     Bx��J*  "          @���@��
��z�@l(�BffC�J=@��
�333@h��B��C��f                                    Bx��X�  
�          @Ǯ@��
�\)@uB=qC���@��
�}p�@p  B(�C�C�                                    Bx��gv  
�          @�{@�33�z�@i��B�C�J=@�33�{@X��B��C�7
                                    Bx��v  �          @�G�@�G��!G�@J=qA�Q�C��@�G��7�@6ffA؏\C��=                                    Bx����  
�          @���@����K�@:�HA�z�C�u�@����_\)@!�A�
=C�+�                                    Bx���h  T          @��@��R�]p�@333A�33C�
@��R�p��@Q�A�C��                                    Bx���  T          @�\)@�(��ff@z�A���C�\)@�(��ff@�
A�z�C�1�                                    Bx����  
�          @���@�G��G�?�A��\C���@�G��{?ǮAf�\C��
                                    Bx���Z  
�          @ȣ�@���#33?�=qA��
C�\)@���0  ?�G�A_�
C���                                    Bx���   �          @�\)@����N{?��
A���C���@����Z=q?���AMp�C��                                    Bx��ܦ  T          @�@���P��?�=qAm��C�k�@���[�?�
=A0Q�C�                                    Bx���L  "          @�
=@���O\)?�
=AT��C��3@���XQ�?��
Az�C�=q                                    Bx����  �          @�  @�z��=p�?��A ��C�� @�z��Dz�?8Q�@�(�C�.                                    Bx���  
�          @�@�  �7�?�ffAh  C��f@�  �B�\?�
=A0��C���                                    Bx��>  
�          @�p�@�\)�9��?\Ac�C���@�\)�C�
?�33A+�C��
                                    Bx��%�  �          @��@�{�9��?�{As\)C�h�@�{�Dz�?��RA:�\C���                                    Bx��4�  �          @��@�33�8Q�?��A�33C�K�@�33�E?\Ad��C�g�                                    Bx��C0  
�          @�
=@��:=q?��
A�33C�W
@��G
=?�33APz�C���                                    Bx��Q�  �          @��H@��H���H?�@�(�C�*=@��H��\>�33@H��C��=                                    Bx��`|  
�          @��H@��\�*=q?��AC\)C�>�@��\�3�
?}p�A  C���                                    Bx��o"  "          @���@�{�1G�@�A��C�T{@�{�C33@   A�{C�                                      Bx��}�  T          @�p�@qG��g
=@P��B (�C��@qG���  @1G�A�Q�C���                                    Bx���n  T          @���@hQ��n�R@^�RB(�C��@hQ�����@>{A��C���                                    Bx���  
�          @Ǯ@c33�s33@W�B�C���@c33��ff@6ffAڸRC��                                    Bx����  
f          @�ff@~�R�\(�@Q�A�G�C��
@~�R�u@333AָRC�f                                    Bx���`  �          @�p�@|(��n�R@;�A�
=C�AH@|(����H@=qA�Q�C���                                    Bx���  T          @���@b�\����@AG�A��HC��@b�\��z�@p�A�Q�C�u�                                    Bx��լ  
�          @���@J=q����@L��A�(�C���@J=q��G�@'�Aʣ�C�k�                                    Bx���R  T          @ƸR@�ff�_\)@=p�A�C�%@�ff�w
=@{A��C���                                    Bx����  �          @ȣ�@u��mp�@N{A�
=C��{@u����@,(�A���C�|)                                    Bx���  �          @ʏ\@�Q��p  @2�\AхC�U�@�Q����H@��A�G�C��                                    Bx��D  T          @��H@���P  @<��A�Q�C�H�@���g�@�RA���C�˅                                    Bx���  �          @ȣ�@���L��@E�A��
C��@���e@'
=A�\)C�q�                                    Bx��-�  T          @�G�@�\)�Fff@HQ�A��C���@�\)�`��@+�A�p�C��q                                    Bx��<6  
�          @�\)@���B�\@J�HA��RC���@���\��@-p�A���C���                                    Bx��J�  �          @�{@�\)�B�\@@��A��C��@�\)�[�@#�
A�
=C�AH                                    Bx��Y�  
�          @ƸR@����I��@5A�(�C���@����aG�@�A�C�
=                                    Bx��h(  �          @ʏ\@���>�R@5�A�Q�C��@���Vff@Q�A���C���                                    Bx��v�  T          @���@�{�8��@%�A�=qC��R@�{�N{@��A��
C�n                                    Bx���t  T          @ʏ\@���P  @{A�=qC��R@���dz�?�p�A�=qC��{                                    Bx���  T          @�G�@���mp�@!G�A���C��{@������?��HA�33C��                                    Bx����  
�          @�33@�{�mp�@(Q�A�p�C��@�{����@�
A��C��                                    Bx���f  	�          @��@~{�g
=@^{B��C�� @~{��=q@:=qA���C�#�                                    Bx���  �          @�
=@s�
�U�@���B��C�p�@s�
�xQ�@_\)B�RC�@                                     Bx��β  
�          @Ϯ@q��S�
@�33B�
C�e@q��xQ�@dz�B�\C�#�                                    Bx���X  
Z          @�  @q��QG�@���B��C��
@q��vff@g�B�\C�E                                    Bx����  z          @�Q�@k��<��@��RB-�\C���@k��e�@~{B�HC��f                                    Bx����  �          @���@x���\)@�  B7(�C���@x���J�H@�=qB#  C�n                                    Bx��	J  
�          @׮@�����@�G�B(�C�U�@���Fff@��
BffC�G�                                    Bx���  "          @�33@���
=q@��B;�C�L�@���8��@��B)(�C���                                    Bx��&�  
Z          @�@�z��{@�  BD�C��=@�z��(Q�@��B4
=C��f                                    Bx��5<  
(          @�\)@�33�˅@��RBL��C�ff@�33���@���B=G�C��                                    Bx��C�  �          @�p�@�z���  ��{�^�HC�\)@�z���ff��
���\C�L�                                    Bx��R�  
�          @�=q@�=q��33��Q��n�RC�@�=q�����=q���HC���                                    Bx��a.  �          @�\)@�(����H��33�#�
C�>�@�(���33�������C��{                                    Bx��o�  �          @��
@L����(�������C�e@L�����Ϳ�33����C���                                    Bx��~z  �          @�=q@z�H���
�������C�k�@z�H��\)��(��4Q�C��\                                    Bx���   �          @�  @�33��\)<#�
=uC�{@�33��p��333��p�C�B�                                    Bx����  �          @ƸR@�ff���׿   ��C�n@�ff��zῚ�H�4��C�޸                                    Bx���l  �          @�\)@�z����;�z��(��C���@�z���G����\��HC�(�                                    Bx���  �          @�\)@�=q��{���R�5�C�y�@�=q���\��ff��
C��\                                    Bx��Ǹ  �          @�33@��
��\)�k���C�N@��
��z�s33�33C��)                                    Bx���^  �          @���@y����������O\)C��)@y����������"�\C��                                    Bx���  �          @�33@l(����׾�z��5C�� @l(�������\� z�C��
                                    Bx���  �          @�\)@n{���Ϳ����RC�P�@n{��  ���\�C�
C�                                    Bx��P  T          @�ff@�  ����=���?uC�q�@�  ��Q�(���C���                                    Bx���  �          @�=q@r�\��zᾔz��9��C�h�@r�\���ÿ�G�� z�C�                                    Bx���  �          @��@}p���p�?�@��\C��{@}p���{�W
=�G�C�޸                                    Bx��.B  �          @��R@�������?��\AF=qC��@������?\)@��RC���                                    Bx��<�  �          @�p�@�  ��Q�?�R@\C�S3@�  �����u�
=qC�(�                                    Bx��K�  �          @��
@mp����R��=q�'�C�޸@mp���33��G�� (�C�7
                                    Bx��Z4  �          @��@qG����R�u���C��@qG�����}p��z�C�k�                                    Bx��h�  �          @�@p����  ��
=��C���@p�����
�����6�HC�7
                                    Bx��w�  �          @�p�@p  �xQ켣�
�L��C�H@p  �tz�.{��
=C�:�                                    Bx���&  �          @�
=@hQ����H��Q�}p�C�˅@hQ���Q�J=q�G�C�\                                    Bx����  �          @���@r�\��Q�.{��G�C��{@r�\�z�H�Y����\C�f                                    Bx���r  �          @�G�@a���Q�L�Ϳ��C��@a���{�J=q��HC��                                    Bx���  T          @�G�@o\)��  >��R@N{C��H@o\)��  �Ǯ���\C���                                    Bx����  T          @��@fff���?z�@��C�u�@fff��{�#�
��{C�T{                                    Bx���d  �          @�Q�@Z�H��=q?#�
@�33C�j=@Z�H����.{��p�C�K�                                    Bx���
  �          @�=q@(����(�?!G�@�
=C��R@(����p���\)�2�\C��f                                    Bx���  �          @��\@H�����
?
=@��C�j=@H����zᾊ=q�,(�C�W
                                    Bx���V  �          @�
=@0  ��
=?\)@�33C�|)@0  ��������UC�p�                                    Bx��	�  �          @�\)@�R���H?�{A4Q�C��\@�R���R>8Q�?���C��3                                    Bx���  �          @�p�@ff����?���A/
=C��@ff��(�>#�
?�=qC�G�                                    Bx��'H  �          @��\@(�����?Q�Az�C�aH@(����(��L�Ϳ�C�5�                                    Bx��5�  �          @�Q�@5���z�>�ff@���C��3@5���z�Ǯ��33C��                                    Bx��D�  T          @�z�@�ff��G�=#�
>�G�C���@�ff��G��#�
���C���                                    Bx��S:  �          @��@��\�u���R�y�C��3@��\<���G��|z�>�33                                    Bx��a�  �          @��@��ý��\�uG�C�b�@���>#�
��G��t��?�z�                                    Bx��p�  �          @�=q@��#�
��G��G\)C�5�@�=��
��G��H��?E�                                    Bx��,  �          @�\)@��þ��ÿ�Q��h��C�J=@��ýu��(��o
=C���                                    Bx����  �          @�  @��Ϳ�R�(����(�C��@��;��B�\���C��\                                    Bx���x  �          @���@�Q쾨�ý�G����C�` @�Q쾙���.{��33C��H                                    Bx���  S          @���@��?!G����
�#�
@���@��?�R=���?�G�@�ff                                    Bx����  �          @���@��H>�p��h���  @tz�@��H?
=q�Tz���@��                                    Bx���j  �          @��@���>�
=��p��o�@�(�@���?333�����]�@�p�                                    Bx���  �          @�{@�G��
=�n{�  C���@�G����Ϳ��
�'�C��                                    Bx���  �          @�(�@��\>��ÿ�R���H@W
=@��\>�G�������H@�                                      Bx���\  �          @��H@�G���R�L����
C��\@�G��녾��
�O\)C�\                                    Bx��  �          @���@�\)�����{�^�RC��f@�\)����G���33C�J=                                    Bx���  
�          @�
=@���O\)=��
?L��C��q@���O\)���
�Y��C��q                                    Bx�� N  �          @��R@�p��+�=u?��C�}q@�p��+���\)�B�\C��                                     Bx��.�  �          @�p�@��;�33=�\)?J=qC�"�@��;�Q�<#�
=uC�R                                    Bx��=�  �          @�p�@�z�>8Q�>���@�\)?�z�@�z�=���>�(�@���?�ff                                    Bx��L@  �          @�p�@��>�?�@�(�@��@��>�{?(�@�
=@j=q                                    Bx��Z�  �          @�ff@�\)?��>��@�\)A]�@�\)?�?5@�=qAI                                    Bx��i�  �          @��R@�p�?�
=?^�RA�HAw
=@�p�?�p�?���A@(�AU�                                    Bx��x2  �          @�{@�
=?n{?�{A>{A"{@�
=?0��?��
AZ=q@�=q                                    Bx����  �          @��R@�Q�?8Q�?��HAL��@���@�Q�>�?���AaG�@�G�                                    Bx���~  �          @��R@�z�?G�?�A��RA	�@�z�>�(�?�ffA��@���                                    Bx���$  �          @�@��R?Tz�?�z�A�Q�A\)@��R>�(�@�A��@�                                    Bx����  T          @��@�  ?�z�?�Q�A���AQ�@�  ?L��?��A���A�                                    Bx���p  �          @�@�?�?�\)A@z�A��@�?�=q?�(�A}��A�                                    Bx���  �          @��@���?�{?�  A��\A�=q@���?�(�@�\A�
=AdQ�                                    Bx��޼  �          @��@�Q�?�33?�z�A���A��@�Q�?��
?�z�A���AD(�                                    Bx���b  �          @�\)@��?&ff?333@��@�\)@��>��H?Q�A��@�                                      Bx���  �          @�@�=q?��?O\)AQ�@�G�@�=q>�
=?k�A#�@�G�                                    Bx��
�  �          @��@�?�G�?���A��A��@�?�
=?�p�A�p�Ab�\                                    Bx��T  �          @�p�@���?��H?��AhQ�A�  @���?�z�?ǮA�p�AZ�H                                    Bx��'�  �          @�{@�{?O\)�
=��\)AQ�@�{?h�þ����A'33                                    Bx��6�  �          @��@��
���	����{C�J=@��
>��
�Q���=q@fff                                    Bx��EF  �          @��H@��R��\)� ������C���@��R>�33��p���(�@s�
                                    Bx��S�  �          @�z�@����\)��
��p�C�t{@��>���z�����?�Q�                                    Bx��b�  �          @��@�p��@  �
=q���C��f@�p���=q�G���p�C�z�                                    Bx��q8  �          @�@�\)�G�����Q�C��H@�\)�����	�����RC�+�                                    Bx���  �          @��@��þ.{�h���C�#�@���<��n{�(�>��
                                    Bx����  T          @�(�@��\�����
=��  C�|)@��\�#�
�#�
��Q�C�(�                                    Bx���*  �          @�(�@��H>��>�p�@r�\@'�@��H>.{>�
=@�Q�?�G�                                    Bx����  �          @���@���?G�?Y��A	�A   @���?z�?}p�A"{@�
=                                    Bx���v  �          @�(�@�
=?n{?k�AffA�\@�
=?5?���A3�
@�p�                                    Bx���  �          @�(�@��?333?ٙ�A��H@�G�@��>��
?�A�z�@\��                                    Bx����  �          @�z�@��>W
=?�33A�z�@G�@�����?�33A��C�5�                                    Bx���h  �          @�33@�Q�?0��?���A�z�@�  @�Q�>���?��HA�(�@dz�                                    Bx���  �          @�=q@�p����
?�\)A��C�� @�p���(�?���A��RC��H                                    Bx���  �          @�33@�33>�?���Ar{@���@�33>��?\A~{?��                                    Bx��Z  �          @���@�(�>��H?�(�A���@��@�(�=L��@�A��H?
=q                                    Bx��!   �          @�z�@�
=?�G�?��A���A.�H@�
=?�@z�A��@ƸR                                    Bx��/�  �          @���@�=q?�=q?�Q�A��
A@��@�=q?!G�@Q�A���@�\                                    Bx��>L  �          @�=q@��?��@�A��AB�R@��?(�@�RA��\@�p�                                    Bx��L�  T          @�(�@�  ?.{?J=qA{@ᙚ@�  >��H?k�A�
@�=q                                    Bx��[�  �          @��@��?���?�Q�AhQ�AH  @��?Y��?�A�p�A�H                                    Bx��j>  �          @��R@�\)?�G�?xQ�A  AO�@�\)?�G�?�(�AF=qA'�                                    Bx��x�  �          @�  @�33?�{?�  AH��A�(�@�33?��?���A�AYG�                                    Bx����  �          @�p�@��?�33?!G�@ϮA�Q�@��?�(�?}p�A"�RAw\)                                    Bx���0  �          @�{@��H?W
=��ff��=qA  @��H?k���  �#33A��                                    Bx����  �          @��R@�z�?aG�>8Q�?��AG�@�z�?O\)>�p�@n�RA33                                    Bx���|  �          @�{@��?(���������@�p�@��?.{��  �!G�@�p�                                    Bx���"  �          @��@�녿+����
��33C�c�@�녾�  �����ffC���                                    Bx����  �          @�p�@���+���Q�����C�q�@����=q��ff��  C���                                    Bx���n  �          @�{@�=q�8Q�����C�
=@�=q>����\)����@1�                                    Bx���  �          @��@��H>��Ϳ�z��;�@��\@��H?(�ÿ���'�@�{                                    Bx����  �          @��R@��
�W
=�n{��\C���@��
<#�
�u��\=�\)                                    Bx��`  �          @�p�@��H��Ϳ=p���G�C�1�@��H��Q�Y���
{C�%                                    Bx��  �          @�{@��ÿW
=����+33C���@��ÿ녿�(��F�\C�
=                                    Bx��(�  �          @��R@�\)���׿�=q�-�C�*=@�\)�Y������TQ�C��
                                    Bx��7R  �          @���@��ÿz��G��s\)C��@��þW
=������\)C��=                                    Bx��E�  �          @�G�@�녾k���
=�g
=C��{@��=�G������i�?�
=                                    Bx��T�  �          @���@��
��
=�����@��C���@��
�����\�K\)C�Z�                                    Bx��cD  �          @�=q@��ÿY�������{
=C��)@��þ���p���z�C��                                     Bx��q�  �          @�z�@�=q�8Q��p���G�C�U�@�=q��z�������C���                                    Bx����  �          @���@�  =�G���(���33?���@�  ?�\��33����@��H                                    Bx���6  �          @�G�@�zῇ������33C�j=@�z�#�
������z�C��R                                    Bx����  T          @���@�33���ÿ�=q�S�C��q@�33������(���{C�k�                                    Bx����  �          @���@�ff��R��{�1��C�b�@�ff��33��{���
C��{                                    Bx���(  �          @�@�=q�ff�}p�� ��C���@�=q��\�\�y�C�f                                    Bx����  �          @�(�@����׿���aG�C�p�@����G���(���\)C�aH                                    Bx���t  �          @��@����
=��z��E��C�G�@���˅�˅����C��                                    Bx���  �          @��R@�
=���R����4  C��@�
=��
=�\����C���                                    Bx����  �          @�p�@�\)�����xQ����C��f@�\)��zῴz��g33C�5�                                    Bx��f  �          @�33@�z��녿�
=�A�C���@�z��ff�������HC��)                                    Bx��  �          @��@����(Q�z�H�)C��@�����
�˅��33C�y�                                    Bx��!�  �          @�\)@����K��L�����C��3@����8�ÿ�ff��Q�C�E                                    Bx��0X  �          @�=q@7
=���H��(���(�C�xR@7
=�w
=����p�C�AH                                    Bx��>�  �          @�z�@��������
=C��R@���ÿ�����\C�S3                                    Bx��M�  �          @���@5���
��R���C�~�@5���\��
=��p�C�ff                                    Bx��\J  
�          @�33@1G���녿0���ᙚC���@1G������������C��)                                    Bx��j�  �          @�33@5���G�����C�N@5���Q�ٙ�����C�3                                    Bx��y�  �          @�(�@c�
���H�Ǯ�|��C���@c�
��33�����o�C�y�                                    Bx���<  �          @��@O\)���
��33�c33C�� @O\)��z῾�R�t��C�33                                    Bx����  �          @���@HQ�����Q��l��C���@HQ���{�\�zffC��
                                    Bx����  �          @�33@:=q��Q������C���@:=q��
=�ٙ����C��                                    Bx���.  �          @�33@'�����z����RC�@'������ff��C�˅                                    Bx����  �          @��H@1����ÿ@  ���RC��@1���{��
=���C��                                    Bx���z  �          @�33@X����{�����EC��@X����
=��z��j{C�b�                                    Bx���   �          @��
@>{��  ��=q�/\)C�  @>{���ÿ��H�qC��                                     Bx����  �          @��@B�\���׽�G�����C�C�@B�\���H�����W33C�Ǯ                                    Bx���l  �          @�@'�����    �#�
C���@'�������
�QG�C�%                                    Bx��  �          @�@'���G�>#�
?�z�C���@'���������7
=C�                                    Bx���  �          @�p�@!G�����?�@�Q�C�AH@!G����׿=p���Q�C�N                                    Bx��)^  �          @�p�@-p����R>�{@[�C�Q�@-p���(��p�����C���                                    Bx��8  �          @��@<(����>�p�@l��C��R@<(���  �aG���
C��                                    Bx��F�  �          @�z�@[���
=>L��@�\C��)@[����
�u���C��                                    Bx��UP  �          @��@���w
=��Q�k�C�=q@���l�Ϳ�=q�2=qC��3                                    Bx��c�  �          @��@xQ��~�R���
�G�C�"�@xQ��u������733C��3                                    Bx��r�  �          @�33@����w����Ϳ��\C�3@����n{�����5C���                                    Bx���B  �          @���@����^�R�����=qC��)@����N�R��{�_�C��)                                    Bx����  �          @��H@��\�a녽u�(�C�n@��\�X�ÿz�H�!�C��)                                    Bx����  �          @��@�33�mp��\�xQ�C���@�33�^�R��=q�^ffC��
                                    Bx���4  �          @��@���3�
�(���(�C�p�@���#33�����_�
C��H                                    Bx����  �          @��R@�ff���R�����5p�C�@�ff��녿Ǯ����C��
                                    Bx��ʀ  T          @�@��\��녿�G���{C�U�@��\��������C���                                    Bx���&  T          @��@�녿�\)��{���RC�p�@�녿c�
�����C��q                                    Bx����  �          @���@��׿��ÿ����C��)@��׿J=q�����C���                                    Bx���r  �          @��\@z�H�y��>��@��RC��@z�H�w��#�
����C���                                    Bx��  �          @�G�@p���i��?
=@�ffC��\@p���k�������=qC��{                                    Bx���  �          @�(�@:�H����?�Q�Axz�C�  @:�H��Q�>�  @(��C�p�                                    Bx��"d  �          @�p�@G
=��Q�?�p�AQ�C��@G
=��=L��?�\C��f                                    Bx��1
  �          @�z�@@������?�z�AF�RC�~�@@����ff�#�
�ǮC��                                    Bx��?�  �          @�(�@3�
��Q�?�(�AV�HC��R@3�
��<�>�Q�C�4{                                    Bx��NV  �          @��@
=q���?���A��\C��q@
=q���?��@�C��                                    Bx��\�  �          @�G�?�������@   A�=qC�)?�����z�?5@�\)C�W
                                    Bx��k�  �          @��\@6ff����?��\A]G�C���@6ff���=u?.{C�8R                                    Bx��zH  �          @�G�@+�����?�(�A�G�C�XR@+���=q>�G�@�G�C��                                    Bx����  T          @�
=@/\)��(�?�(�A�
=C���@/\)��{>��@���C�#�                                    Bx����  �          @�  @H����?޸RA���C�s3@H����  ?
=q@��HC�w
                                    Bx���:  �          @���@QG���z�?�\)A�z�C�%@QG���>�(�@�\)C�:�                                    Bx����  �          @��@c33�|(�?n{AffC�  @c33�����8Q����C��H                                    Bx��Æ  �          @��
@]p���Q�?Y��A{C�b�@]p����H��\)�=p�C��                                    Bx���,  �          @�p�@o\)�u�?=p�@��\C�%@o\)�x�þ�{�g
=C��                                    Bx����  
�          @��@\)�r�\?+�@޸RC�=q@\)�u��Ǯ���C�R                                    Bx���x  �          @��\@s33�xQ콣�
�Q�C�+�@s33�mp�����B=qC�Ф                                    Bx���  �          @�=q@aG������\)�:�\C�L�@aG��h�������(�C��                                    Bx���  T          @��@J�H����>�Q�@l(�C�p�@J�H��
=�h���
=C���                                    Bx��j  �          @�=q@5�����?�G�A|(�C���@5�����>L��@G�C�Q�                                    Bx��*  �          @�z�@Fff��  ?�  ANffC�Ff@Fff��p��#�
��ffC�Ǯ                                    Bx��8�  �          @��
@-p����?s33A�
C���@-p���z��(���33C���                                    Bx��G\  �          @���@8������?8Q�@��
C��=@8����녿#�
�ϮC��H                                    Bx��V  �          @��
@(����?E�@�z�C�@(���  �&ff���C�                                    Bx��d�  �          @�z�@!���  ?0��@�=qC�l�@!�����:�H���C�o\                                    Bx��sN  �          @�33@5���Q�?�@�=qC�aH@5���\)�J=q�ffC�w
                                    Bx����  _          @�33@;����>B�\?��RC�� @;���33����=�C�@                                     Bx����  I          @�=q@<����p�>W
=@p�C�!H@<����G���{�8��C�~�                                    Bx���@  _          @���@p���?�ADz�C�{@p���녾���+�C���                                    Bx����  �          @�=q@7
=��Q�>k�@
=C�}q@7
=��(���\)�:ffC���                                    Bx����  �          @���@8Q����R>L��@�C���@8Q���=q����>�HC�                                      Bx���2  
�          @��H@s�
�~{�\(����C��@s�
�dz��������C�l�                                    Bx����  �          @�=q@p  �y�����
�UG�C��=@p  �XQ���ƸRC���                                    Bx���~  
�          @��H@w��}p��\)��G�C�/\@w��hQ��z���
=C�ff                                    Bx���$  �          @�=q@�z��W��8Q��\)C�9�@�z��K������<��C�f                                    Bx���  �          @��\@^{�mp��
=��  C��=@^{�=p��E����C�Ǯ                                    Bx��p  �          @�p�@h���dz�����ȣ�C�@h���/\)�S�
��HC�~�                                    Bx��#  �          @�{@qG��vff��ff����C�0�@qG��J�H�4z���z�C��                                    Bx��1�  �          @�@}p��\)�0���߮C�h�@}p��g���ff��z�C�Ǯ                                    Bx��@b  �          @��R@~{�\)?��@��HC�j=@~{�\)��R�ȣ�C�t{                                    Bx��O  �          @�  @dz����?޸RA���C�K�@dz����>��@��C�=q                                    Bx��]�  �          @��@QG���G�@�A��C�}q@QG���  ?k�A=qC��                                    Bx��lT  �          @��H@h���l��@G�A�Q�C�J=@h�����
?^�RA�C���                                    Bx��z�  �          @�Q�@aG��c�
@\)A���C�^�@aG�����?�{A;�C���                                    Bx����  �          @�(�@1G���Q�@{AîC�S3@1G���\)?uA%��C�޸                                    Bx���F  �          @��@!G���  @�A��HC�W
@!G����?5@�p�C�8R                                    Bx����  T          @�z�@#�
��z�?�\A��
C�"�@#�
���R>�
=@�  C�B�                                    Bx����  �          @��@*=q����?У�A�ffC���@*=q��>�\)@<(�C��=                                    Bx���8  �          @�z�@'��{�@��A¸RC��{@'���z�?fffA�C�l�                                    Bx����  �          @�ff@,(���녿�Q���\)C��)@,(��c�
�G
=�C��                                    Bx���  �          @�(�@z��P  �g
=�(��C��=@z��33�����[�\C�K�                                    Bx���*  �          @�ff@�R�^{�aG��!��C�W
@�R��\��(��W�C��                                    Bx����  �          @���@(��`���O\)�\)C�h�@(��=q���
�H�HC��3                                    Bx��v  �          @��@�H�L(��l���+33C���@�H���H��\)�\C��                                     Bx��  �          @�33@   �o\)�S33�33C��q@   �&ff��  �F�HC��                                    Bx��*�  �          @��@&ff���\��p���C���@&ff�w��@���Q�C��                                    Bx��9h  �          @��R@"�\���׽�Q�c�
C�p�@"�\���׿Ǯ��
=C��                                    Bx��H  �          @��R@4z���Q�����
=C�S3@4z����
��p����HC�k�                                    Bx��V�  �          @��H@,(�����?�A9��C�R@,(����;�p��hQ�C�Ф                                    Bx��eZ  �          @��@���z�?�  AuG�C�1�@����
���
�B�\C���                                    Bx��t   �          @��
>��
���\@0  A���C�� >��
��p�?�Q�AFffC���                                    Bx����  �          @��\�E����
@ ��A�C��׿E�����?s33A�C���                                    Bx���L  �          @��\��R��Q�@0  A���C�C׿�R���?��HAJ�RC��                                    Bx����  �          @��R?.{����?�(�Av�RC���?.{��  �.{���C��f                                    Bx����  �          @���?�  ��G�?n{Az�C�^�?�  ��녿E����C�XR                                    Bx���>  �          @��?�  ��
=?��AX��C�5�?�  ���
��33�c33C��                                    Bx����  �          @�p�?
=����?��A1��C��?
=����(���
=C��q                                    Bx��ڊ  �          @���?8Q���G�?333@�p�C��\?8Q�����}p��!�C��
                                    Bx���0  �          @���?+���\)?�33A<Q�C�� ?+���=q�����=qC�q�                                    Bx����  �          @��?L����G�?�A�G�C�Q�?L�����H>�?�{C�
                                    Bx��|  �          @�ff?B�\��?��
Ay�C��
?B�\��(��8Q���
C���                                    Bx��"  �          @�z�>��
��33@�A��C���>��
��=q?8Q�@�z�C��H                                    Bx��#�  �          @��\<#�
���
@+�A�\C��<#�
��ff?��A4(�C��                                    Bx��2n  �          @��\?
=q��G�@�A�p�C��?
=q��  ?&ff@�{C�Ф                                    Bx��A  �          @���
=q���R@p�AхC��q�
=q��
=?W
=A
�HC�'�                                    Bx��O�  �          @�{�h�����@-p�A��
C��R�h����  ?��A1�C�Ff                                    Bx��^`  �          @��Ϳ�p���Q�@'�AݮC|���p����\?�ffA,Q�C~p�                                    Bx��m  �          @������33@#�
Aי�C~�Ϳ������?xQ�Ap�C�f                                    Bx��{�  �          @����p�����@,(�A�RC|���p����?�{A5p�C~�{                                    Bx���R  �          @�zΐ33��{@7
=A�p�C����33���H?�ffAVffC���                                    Bx����  �          @��R?������
@  A��C�33?������?z�@��HC���                                    Bx����  �          @�
=@*�H����>k�@z�C�E@*�H������
�T  C��\                                    Bx���D  �          @��@N{��ff?\)@��HC�1�@N{���Ϳh�����C�Y�                                    Bx����  T          @���@i����>��
@L(�C��)@i����=q��ff�(z�C��                                    Bx��Ӑ  �          @��
@����xQ쿀  ���C���@����XQ�������C��3                                    Bx���6  �          @���@�=q�U���Q���Q�C�.@�=q�%��7���33C��
                                    Bx����  �          @�33@�G��X���G���C��3@�G��'
=�=p���{C�U�                                    Bx����  T          @��H@����G
=�����
=C��@����{�O\)�Q�C�XR                                    Bx��(  �          @���@�p��tz���H���\C���@�p��G
=�333��z�C���                                    Bx���  �          @��@~{�}p������{C���@~{�L(��@����z�C��)                                    Bx��+t  �          @�@�G��q����33C�p�@�G��=p��I��� C��                                    Bx��:  �          @�\)@����p�@%�A��HC���@�����?��\A�C��f                                    Bx��H�  �          @��?�������@O\)B=qC�
?������?�Q�A��C��
                                    Bx��Wf  �          @�p�@���Q�@1G�A��C�J=@���(�?�z�A5�C��                                    Bx��f  T          @�p�@�
��
=@:�HA��C�Ǯ@�
��z�?���AMC�q�                                    Bx��t�  
�          @�
=?�
=���
@l��Bp�C���?�
=����@
�HA�ffC�Ф                                    Bx���X  �          @��@	����\)@Q�B�C���@	������?�p�A�G�C��                                    Bx����  �          @�@�\���@HQ�B �C�@�\���?ǮAt��C��                                    Bx����  �          @�ff?�p���Q�@b�\B��C��?�p���z�?��HA���C���                                    Bx���J  �          @��R?޸R����@O\)B�C�J=?޸R��p�?У�A�C��=                                    Bx����  �          @�Q�?��\��
=@s33B
=C��?��\��@{A�\)C��{                                    Bx��̖  �          @�  >�����{@c33B{C��>������?�z�A��C���                                    Bx���<  �          @�Q�?����p�@�Q�B(p�C�!H?�����R@ ��AƏ\C�8R                                    Bx����  �          @��?@  ��33@~�RB(�C���?@  ��(�@�HA�Q�C��)                                    Bx����  �          @�33?�33��(�@mp�B=qC�` ?�33��=q@	��A��\C�q                                    Bx��.  �          @�z�?����\)@eB�HC�O\?����(�@   A��HC��)                                    Bx���  T          @�  ?�����H@tz�B&��C�� ?�����\@ffA��HC��                                    Bx��$z  T          @�
=?xQ�����@|��B.z�C��f?xQ����@�RA��C�l�                                    Bx��3   
�          @��R?Y������@{�B.G�C��3?Y�����\@p�Ạ�C���                                    Bx��A�  �          @��R?aG��~�R@\)B1��C�=q?aG�����@"�\A�\)C���                                    Bx��Pl  �          @�
=?�ff�~{@~{B0G�C�n?�ff��Q�@!G�A�C��                                    Bx��_  �          @�ff?�  �s�
@��B8�C�Q�?�  ���@,��A�RC���                                    Bx��m�  �          @�\)?�(��x��@z�HB-�C�W
?�(���p�@   Aϙ�C�K�                                    Bx��|^  �          @�{?����p�@X��B(�C�*=?������?�33A�(�C�G�                                    Bx���  �          @�  ?������@QG�B
�C�1�?������
?޸RA�G�C�p�                                    Bx����  �          @��?�33��  @z=qB%�C��3?�33����@��AÅC�XR                                    Bx���P  �          @��?���w�@�G�B,�
C�?����ff@'
=A�(�C�s3                                    Bx����  �          @�33?�=q���@s�
B#��C���?�=q��33@z�A��C��R                                    Bx��Ŝ  �          @��@
=�X��@�  B;
=C���@
=���@=p�A�C�k�                                    Bx���B  T          @�=q?��<(�@�BX�C�Ǯ?�����@`  B�C��=                                    Bx����  �          @�33?�\)�j=q@�\)B>  C���?�\)��z�@EA�ffC�q�                                    Bx���  
�          @��
?�z��qG�@���B8�C�g�?�z���
=@>�RA�C�|)                                    Bx�� 4  �          @��
?˅���\@�Q�B0�C�� ?˅��
=@/\)A�
=C�t{                                    Bx���  �          @�33?�p����
@~�RB#(�C�Z�?�p����@��A�Q�C��f                                    Bx���  �          @��
?�Q�����@tz�B��C���?�Q�����@�A��HC�P�                                    Bx��,&  �          @Å?�����ff@w�BffC���?�����ff@��A�\)C�,�                                    Bx��:�  �          @�(�?�\)���R@j=qB{C��?�\)��(�?�(�A�ffC��{                                    Bx��Ir  �          @���?�p���  @g�B
=C���?�p����?�z�A���C�N                                    Bx��X  �          @���?��
��
=@h��B33C��)?��
��(�?���A��C���                                    Bx��f�  �          @��?�G���=q@`��B�C���?�G���{?��A�
=C�n                                    Bx��ud  �          @�z�?�=q��33@HQ�A��
C�U�?�=q���\?��AI�C�q�                                    Bx���
  �          @�z�?��R��\)@=p�A�RC��?��R���?���A*{C���                                    Bx����  �          @��
?��
����@5A�
=C�Ф?��
����?�  A
=C�)                                    Bx���V  �          @��?������@6ffA܏\C��
?����?�  A�C�"�                                    Bx����  
�          @�{?��
��  @K�A��C���?��
��  ?�z�AS\)C��{                                    Bx����  �          @�ff?��
����@HQ�A�=qC��3?��
��G�?��AH(�C���                                    Bx���H  �          @�ff?�����@333A��HC�%?���Q�?k�Az�C���                                    Bx����  �          @�ff?�33��\)@)��A���C��\?�33��G�?@  @߮C�e                                    Bx���  �          @�ff?xQ���Q�@+�Ạ�C���?xQ��\?E�@�33C��                                    Bx���:  �          @�{?�G�����@"�\A�{C�#�?�G���=q?�R@���C��{                                    Bx���  �          @�{?�=q���\@��A��\C�}q?�=q��=q?�@�(�C�                                    Bx���  �          @�
=?�{���@\��B��C��q?�{���\?�z�Aw�
C���                                    Bx��%,  �          @�\)?�  ��Q�@>{A�Q�C���?�  ��{?�\)A&=qC�3                                    Bx��3�  �          @�\)?����(�@ ��A�(�C��f?������?#�
@�{C���                                    Bx��Bx  �          @�\)?�����p�?�ffA�(�C�� ?�����ff����Q�C�}q                                    Bx��Q  �          @�
=?�����(�?���AM��C���?�����Q�
=��\)C�z�                                    Bx��_�  
�          @ƸR?�  ��
=@L��A��RC�h�?�  ��
=?���AH��C��q                                    Bx��nj  �          @Ǯ?����\)@FffA�RC�s3?����ff?�  A8��C��{                                    Bx��}  �          @Ǯ?�=q����@B�\A�33C�\?�=q��\)?�A,��C�C�                                    Bx����  �          @Ǯ?������
@:=qA��
C���?�������?��\AC�/\                                    Bx���\  �          @�Q�?�(���{@5�A�p�C�W
?�(����?k�A�C���                                    Bx���  �          @�  ?��R��{@3�
AծC�ff?��R���?fffA(�C��                                    Bx����  �          @Ǯ?�33��G�@(Q�A�33C�ٚ?�33���H?0��@�(�C�T{                                    Bx���N  �          @�\)?������@(��AȸRC���?���\?5@љ�C�P�                                    Bx����  �          @�\)?�
=���
@(�A�Q�C��?�
=��33>��H@�Q�C�u�                                    Bx���  �          @�
=?�����Q�@!G�A��
C���?�������?��@��C�P�                                    Bx���@  �          @�{?�����@1G�A��C��\?������?^�RAG�C�Y�                                    Bx�� �  �          @�?�{��(�@-p�A�ffC�q?�{��
=?Q�@���C�o\                                    Bx���  _          @��?�G���p�@ ��A���C�Ǯ?�G���{?�R@���C�                                      Bx��2  �          @��?�������@0��A��C�J=?�����(�?fffA�\C�xR                                    Bx��,�  �          @���?޸R��=q@!G�A�(�C��?޸R��33?(��@�p�C�Ff                                    Bx��;~  �          @�@�����@\)A�=qC���@�����?#�
@�
=C���                                    Bx��J$  �          @�{@����@��A���C���@�����?�@�C�޸                                    Bx��X�  �          @�?���33@�HA���C�ٚ?���33?��@�z�C��                                    Bx��gp  �          @�?�ff��z�@$z�A��HC��?�ff��?.{@ə�C�O\                                    Bx��v  �          @�p�?�����G�@-p�A���C�k�?�����z�?W
=@�33C��q                                    Bx����  �          @�{@ff��{@C�
A�33C��=@ff���?��
A?�
C�0�                                    Bx���b  �          @ƸR@!G�����@1G�Aأ�C�� @!G���{?��A z�C�q�                                    Bx���  �          @ƸR?�
=��z�@%A��HC���?�
=��{?0��@�z�C��                                    Bx����  �          @ƸR?У����@#�
A�G�C�Z�?У���ff?(��@�(�C���                                    Bx���T  �          @�\)?�������@5AمC�j=?�����?uAffC��                                    Bx����  T          @�  ?޸R���@��A�G�C���?޸R���>��
@<(�C��                                    Bx��ܠ  �          @�\)?��H��
=@=qA��\C���?��H��ff?   @��HC�                                      Bx���F  �          @ȣ�@�R����?�  A�z�C��{@�R��녾����z�C��                                     Bx����  �          @���@ ����G�@��A��C�h�@ ����ff>���@6ffC��R                                    Bx���  T          @�=q?�
=��{@
=A�{C��f?�
=���>�@�
=C��                                    Bx��8  �          @ʏ\?޸R����?n{A�C��?޸R�����=q��C��                                    Bx��%�  �          @��?��
���?�Q�A/�C��?��
�����Tz����C���                                    Bx��4�  �          @ə�@ff��G�@-p�AˮC�Ф@ff��(�?Tz�@�C��\                                    Bx��C*  �          @ə�@(�����@FffA��
C�Ǯ@(���Q�?��A<��C�h�                                    Bx��Q�  T          @�  @�\����@:�HA�z�C�9�@�\���R?�{A$(�C��                                    Bx��`v  T          @Ǯ@\)��Q�@>{A�33C��@\)��ff?�A,��C��q                                    Bx��o  T          @�  @Q���ff@ffA�=qC��\@Q����H>W
=?�
=C��                                    Bx��}�  
�          @�G�?��
���\?��As�
C�^�?��
������33�N�RC��                                    Bx���h  �          @���?��H��z�?�  A8(�C��?��H��
=�=p����C���                                    Bx���  �          @��?�Q���
=?uAz�C���?�Q���ff����G�C��                                    Bx����  �          @�  ?Ǯ���H@�A�\)C���?Ǯ����>���@@  C�B�                                    Bx���Z  �          @�
=?�(�����@�A�=qC���?�(����R>�33@P  C��                                    Bx���   �          @�ff?�p����@%�A�{C�4{?�p����?5@�Q�C�O\                                    Bx��զ  �          @�\)@����z�@/\)A��HC��q@����  ?h��A�\C�}q                                    Bx���L  �          @�\)@����@-p�A�z�C�k�@�����?^�RA�C�U�                                    Bx����  �          @�
=?���z�@&ffA�(�C��
?���{?333@�ffC��{                                    Bx���  �          @�
=?�=q����@G�A��C�N?�=q���
<��
>W
=C��                                    Bx��>  
�          @�\)?�����@
�HA��C�p�?������>L��?�\)C�
                                    Bx���  T          @�
=?�������@HQ�A�z�C��?�������?���AC�
C��H                                    Bx��-�  T          @ȣ�>����p�?xQ�AffC���>�����Ϳ����"=qC��=                                    Bx��<0  �          @Ǯ@�
��\)@HQ�A�p�C�9�@�
��\)?�=qAE��C��                                     Bx��J�  T          @�?���i��@�p�B9  C��?�����@0  A㙚C�e                                    Bx��Y|  �          @�z�@����p�@�{Bzz�C�!H@���K�@��BE�HC��                                     Bx��h"  �          @�@#�
�^�R@�G�B��C�� @#�
�(Q�@�ffBU�RC��                                    Bx��v�  �          @�@0�׿��H@�Q�Bm33C��@0���Fff@�  B=(�C��                                     Bx���n  �          @�
=@�R��@���Bi=qC��\@�R�n�R@���B/=qC���                                    Bx���  �          @�Q�?�  �#33@�  Bv�HC�H�?�  ��ff@��B1\)C��R                                    Bx����  �          @���?����6ff@��Bt{C�S3?������@�Q�B*�HC���                                    Bx���`  �          @��?����5�@�
=Bq33C��f?������R@�  B)�\C�@                                     Bx���  �          @��?�\�#�
@�\)Bq��C�K�?�\���R@��HB.�C�aH                                    Bx��ά  �          @��H?޸R�.�R@�ffBm�C�=q?޸R���@�  B(�RC��q                                    Bx���R  �          @ʏ\@ff�333@�(�BZ=qC���@ff��=q@{�B��C�=q                                    Bx����  �          @��@E�I��@�\)B5C�1�@E��ff@Mp�A�
=C�c�                                    Bx����  �          @˅@���J�H@�
=BM�\C���@�����
@j=qB�\C���                                    Bx��	D  �          @ʏ\@*�H�g�@��RB2�C�:�@*�H��(�@A�A��C�]q                                    Bx���  �          @˅@1G��\(�@��B7�\C�j=@1G����@K�A�RC�&f                                    Bx��&�  �          @�z�@HQ��(Q�@���BH�RC���@HQ����\@qG�B  C��R                                    Bx��56  �          @ʏ\@1��Y��@���B8�C��@1���ff@L(�A�z�C�O\                                    Bx��C�  �          @˅@(Q��p  @�z�B/
=C���@(Q���\)@:�HA�\)C��=                                    Bx��R�  �          @�33@3�
��Q�@�Q�BffC�~�@3�
��33@�RA���C�h�                                    Bx��a(  �          @��
@.{�~�R@��B#=qC�.@.{���
@(Q�AÅC��q                                    Bx��o�  �          @˅@<(��q�@�B$�\C��3@<(���{@.{Aʣ�C�e                                    Bx��~t  �          @˅@<���P  @��\B8�C�%@<����=q@QG�A�=qC�s3                                    Bx���  �          @�33@ff�|(�@�(�B.�RC�c�@ff����@6ffA��C�=q                                    Bx����  �          @˅?�z���Q�@�=qB+=qC��?�z���@,��Aȣ�C��\                                    Bx���f  �          @�33?���(�@�\)B'  C�:�?�����@$z�A��HC�{                                    Bx���  �          @�z�@\)���\@n{B�
C��)@\)���@
=A�Q�C��R                                    Bx��ǲ  �          @�z�@<����{@]p�B
=C��=@<����=q?���A��C�z�                                    Bx���X  �          @�(�@2�\��\)@dz�B\)C���@2�\����?�A�  C��                                    Bx����  �          @˅@z���33@?\)A�
=C�@ @z���G�?�A)C��                                    Bx���  �          @�33@����@@��A��C���@�����?���A.{C�*=                                    Bx��J  
2          @˅@�\��p�@_\)B��C�Ff@�\��G�?�Q�Av�RC���                                    Bx���  
�          @��
@����@dz�B=qC�,�@����R?�A��C�o\                                    Bx���  �          @��
?�(���G�@g
=B�C�4{?�(���ff?���A���C���                                    Bx��.<  �          @˅?�G����R@~{B��C��?�G���  @{A��
C�^�                                    Bx��<�  T          @�33?�����Q�@�Q�B��C���?�����=q@�A�(�C��H                                    Bx��K�  T          @��H?��H��=q@z�HB(�C�?��H���\@��A�(�C��q                                    Bx��Z.  �          @�=q?W
=����@J�HA�RC�p�?W
=��(�?��RA4��C��=                                    Bx��h�  �          @ʏ\>8Q�����@X��Bp�C��3>8Q��Å?�p�AZ=qC��3                                    Bx��wz  �          @ʏ\�������@c33B��C�lͽ����Å?�z�As
=C���                                    Bx���   �          @˅?�\���@O\)A��\C��=?�\��p�?��A>�HC�W
                                    Bx����  T          @�33?�=q��  @7
=A�z�C�Ф?�=q��z�?k�A�C�"�                                    Bx���l  T          @��H?ٙ����\@��A�{C�u�?ٙ��\?   @��C��                                    Bx���  T          @��H?����
@  A���C���?���G�>���@.�RC�U�                                    Bx����  �          @�(�?�����p�@
=qA���C���?������>W
=?��C��\                                    Bx���^  T          @��
?����(�@�A��C��3?���\>�p�@XQ�C�5�                                    Bx���  T          @�(�@�H���@��A�
=C�*=@�H���?z�@��C�9�                                    Bx���  �          @˅@Q���
=@�A��C���@Q����R?�@�\)C�ٚ                                    Bx���P  T          @˅@�R��\)@�
A�z�C��@�R��>��@l(�C�O\                                    Bx��	�  �          @˅@����(�@��A�33C�@�����?�@�Q�C�                                      Bx���  �          @��
@
=���@�RA�p�C���@
=��(�?(�@��C��                                    Bx��'B  T          @ʏ\?����33@Q�A��C�&f?����=q>�ff@���C���                                    Bx��5�  �          @�=q?\)���R?��HA�G�C��{?\)�ȣ׽�\)��RC���                                    Bx��D�  T          @ʏ\?u��p�@   A�p�C���?u��  ���
�B�\C�aH                                    Bx��S4  �          @ʏ\?�\)��?�\)A��C�Z�?�\)��
=�\)��ffC�q                                    Bx��a�  T          @�=q?�  ��@�A���C�aH?�  ��(�>�p�@U�C��q                                    Bx��p�  �          @˅?�  ��Q�@{A��C�K�?�  ��p�>u@(�C��{                                    Bx��&  �          @�(�?\��=q@��A�z�C�N?\��ff>��?�ffC��                                     Bx����  �          @˅?������\@A��C��R?�����ff=���?fffC��3                                    Bx���r  T          @��
?��R���\@{A���C�H?��R�Ǯ>aG�@�C��                                     Bx���  �          @��
?�p����@	��A��
C�*=?�p���ff>#�
?�p�C��)                                    Bx����  �          @��
?�(����\@A��C�\?�(���{=�G�?uC���                                    Bx���d  �          @�(�?��H��{@&ffA�z�C�H?��H��
=?!G�@�p�C��                                     Bx���
  �          @���?����G�@
=A��C�C�?����  >�p�@W
=C��3                                    Bx���  �          @���?���Q�@!G�A�p�C���?���Q�?�@���C�Ff                                    Bx���V  T          @˅?�{��Q�@�A���C�W
?�{��
=>�G�@���C���                                    Bx���  �          @�=q@333��33@�RA���C�aH@333���
?5@�
=C�9�                                    Bx���  	�          @��H@AG���
=@=qA�ffC��f@AG���\)?0��@��C�n                                    Bx�� H  "          @ʏ\@Dz���p�@ ��A�
=C��@Dz����R?L��@�Q�C���                                    Bx��.�  
�          @��H@HQ����@!G�A�
=C�H�@HQ���ff?O\)@�33C��                                    Bx��=�  
�          @��H@N�R���
@�RA�  C���@N�R����?J=q@��C�q�                                    Bx��L:  
Z          @��H@
=��(�@
=A��C��q@
=��33?�@��RC���                                    Bx��Z�  J          @˅@����H@!G�A�
=C�˅@����
?.{@�z�C��
                                    Bx��i�  
d          @˅@@����\)@#�
A��C��R@@����G�?Tz�@�G�C�H�                                    Bx��x,  "          @�33@aG���{@�A�ffC�p�@aG���
=?O\)@�\C��q                                    Bx����  "          @˅@]p����R@�HA�=qC�+�@]p���\)?J=q@�
=C��q                                    Bx���x  T          @���@q�����@p�A���C���@q����\?c�
@�
=C�Z�                                    Bx���  T          @�(�@Vff��p�@/\)A��
C���@Vff����?�{A Q�C�'�                                    Bx����  T          @��
@L(���\)@5A�p�C��R@L(���z�?�
=A*�HC�K�                                    Bx���j  
�          @˅@hQ���33@8��A�{C���@hQ�����?���ADz�C��                                     Bx���  
�          @�z�@l����z�@
=A��C�C�@l������?B�\@��HC��{                                    Bx��޶  
�          @�z�@[�����@p�A�(�C���@[���33?
=q@��HC�Z�                                    Bx���\  
�          @���@j=q��p�@��A���C��@j=q��{?G�@�G�C���                                    Bx���  "          @���@a���{@"�\A�
=C�w
@a���  ?k�A�
C��\                                    Bx��
�  �          @�p�@_\)���@$z�A�z�C�0�@_\)���?n{Ap�C��=                                    Bx��N  T          @��@N{���H@.{Aȣ�C��\@N{��ff?��A{C�Ff                                    Bx��'�  
�          @�z�@L(����
@(��A�\)C��)@L(����R?uA
�HC�'�                                    Bx��6�  
�          @���@AG���{@,(�A�\)C��{@AG���G�?}p�A�\C�J=                                    