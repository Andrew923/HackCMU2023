CDF  �   
      time          *   Conventions       ACDD-1.3, Spase v2.2.3     title         /DSCOVR Magnetometer Level 2 One Minute Averages    id        Doe_m1m_dscovr_s20230111000000_e20230111235959_p20230112020414_pub.nc   naming_authority      gov.noaa.swpc      program       DSCOVR     summary       }Interplanetary magnetic field observations collected from magnetometer on DSCOVR satellite - 1-minute average of Level 1 data      keywords      _NumericalData.ObservedRegion.Heliosphere.NearEarth, NumericalData.MeasurementType.MagneticField    keywords_vocabulary       Spase v2.2.2   
references        �; DSCOVR TIME SERIES DATA AVERAGES ALGORITHM THEORETICAL BASIS DOCUMENT, v2.4; GSE TO GSM COORDINATE TRANSFORMATION ALGORITHM THEORETICAL BASIS DOCUMENT v2.1      metadata_link         �http://www.ngdc.noaa.gov/docucomp/page?xml=NOAA/NESDIS/NGDC/STP/Space_Weather/iso/xml/satellite-systems_dscovr.xml&view=getDataView&header=none    license       Spase.Access Rights.Open   institution       NOAA   source        DSCOVR Magnetometer Level 1    platform      'Deep Space Climate Observatory (DSCOVR)    
instrument        +boom-mounted triaxial fluxgate magnetometer    history       ,DSCOVR real-time telemetry processing system   	algorithm         FDSCOVR MAGNETOMETER LEVEL 1B DATA ALGORITHM THEORETICAL BASIS DOCUMENT     algorithmVersion      B      algorithmDate         
2015-10-15     processing_level      Level 2    processing_level_description      11-minute average using Hodges-Lehmann M-estimator      date_created      2023-01-12T02:04:14.864Z   date_calibration_data_updated         2022-11-28T00:00:00.000Z   time_coverage_duration        P01D   time_coverage_start       2023-01-11T00:00:00.000Z   time_coverage_end         2023-01-11T23:59:59.000Z   time_coverage_resolution      PT1M   creator_name      Doug Biesecker     creator_type      person     creator_institution       DOC/NOAA/NWS/NCEP/SWPC     creator_email         doug.biesecker@noaa.gov    creator_url       http://www.swpc.noaa.gov/      publisher_name         National Geophysical Data Center   publisher_type        institution    publisher_institution         DOC/NOAA/NESDIS/NGDC   publisher_email       william.rowland@noaa.gov   publisher_url          http://www.ngdc.noaa.gov/dscovr/   records_maximum         �   records_present         �   records_data        �   records_fill         �   records_missing                    time                description       "date and time for each observation     
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
short_name        overall_quality    C_format      %d     units         n/a    lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale        7lBxY�   �          A33@�Q����?��
AB�RC��@�Q���  ����n�RC�4{                                    BxY��  T          A
=@�Q���  @Af�HC��@�Q���녾�����C�J=                                    BxY�#L  �          A�H@u���\)?ǮA-G�C��{@u���33�L������C��H                                    BxY�1�  �          A{@�Q���\?�z�AC���@�Q���p��fff��  C���                                    BxY�@�  "          A�R@q���@]p�A�ffC���@q���?�
=A z�C�xR                                    BxY�O>  
�          A  @xQ���@p��A�(�C�\@xQ��ᙚ?�\)A5��C�h�                                    BxY�]�  �          A��@������@5A�(�C���@����ff?
=@��C��R                                    BxY�l�  �          A��@}p���@<(�A�{C�P�@}p����?.{@�\)C�G�                                    BxY�{0  �          A��@�����?��Az�C�:�@�����
=��{��C�.                                    BxY��  �          A
=@s33��Q�?�R@�33C�@s33��녿�{�F=qC�Z�                                    BxY�|  �          A
=q@p����������*�HC��@p�������2�\���\C�޸                                    BxY�"  
�          A{@������
@w�A�p�C�S3@������H@��Am�C���                                    BxY��  
�          A��@�
=����@�G�A��C��3@�
=���@#�
A�{C���                                    BxY��n  "          Az�@�Q����@=p�A�ffC�f@�Q��ƸR?��@�C�w
                                    BxY��  
�          A��@�
=��
=?���A.�\C��@�
=��z�
=q�o\)C�j=                                    BxY��  "          Ap�@����(�?�AJffC�j=@���ۅ��Q��!G�C��3                                    BxY��`  T          A��@��H�ڏ\@��A}p�C�q�@��H��ff    �#�
C���                                    BxY��  �          A�@�33�ʏ\@�
A�p�C���@�33��\)>W
=?���C��                                    BxY��  
�          Aff@�{�Ǯ@,(�A��C�0�@�{�أ�?
=@�G�C��                                    BxY�R  �          A�@��
��z�@H��A��
C��\@��
�ᙚ?z�H@�(�C��q                                    BxY�*�  
�          A
=@~{�θR@l(�AϮC�Ǯ@~{��G�?�p�A!C�G�                                    BxY�9�  
�          Aff@o\)��{@s33A�ffC�
@o\)���?���A0  C���                                    BxY�HD  
(          A�\@z=q��=q@�  A�  C�ff@z=q��33@��Al��C�n                                    BxY�V�  �          A\)@����ff@���A��C�xR@����  @{Atz�C�\)                                    BxY�e�  T          A33@��R��G�>\@&ffC��@��R��=q��G��A�C���                                    BxY�t6  
�          A
�R@��H��z����j=qC�7
@��H�����!G���33C�]q                                    BxY��  "          A�
@������?�A=G�C�s3@����\)��ff�L��C�                                    BxY㑂  
�          A��@�Q���G�@Q�A�Q�C�Ff@�Q���ff>\)?}p�C��\                                    BxY�(  T          A��@�Q���33?�(�A	�C��3@�Q���z�xQ����HC��                                     BxY��  T          A�
@����=q?�\)@��C��\@����=q������RC��                                    BxY�t  
�          A(�@�ff��  ?���@�{C�|)@�ff��  ��������C�~�                                    BxY��  
�          A��@�\)�ۅ?�Q�A  C��
@�\)��z῀  ��
=C���                                    BxY���  �          A�@��
��ff?�
=A{C�@��
��\)�u���C��3                                    BxY��f  �          AQ�@�ff����?\)@qG�C��{@�ff��\)��{�.�\C�*=                                    BxY��  T          A33@����k��˅C��R@���  �������C���                                    BxY��  �          A��@�=q��{@��Ap��C�j=@�=q��׽#�
���C���                                    BxY�X  �          A��@����G�@0��A�G�C�E@�����H?+�@��C�%                                    BxY�#�  	�          A�@�����ff@
=Ak�
C�AH@�����G�<#�
=��
C��\                                    BxY�2�  �          AG�@�(���z�?��AH��C��3@�(���(���=q��33C�0�                                    BxY�AJ  �          A��@����z�?��AH  C�Ф@����33��
=�;�C�k�                                    BxY�O�  "          A��@�
=�ʏ\?��A)�C���@�
=�Ϯ���H�XQ�C�b�                                    BxY�^�  "          AG�@�
=��
=?�p�A$(�C��@�
=�Ӆ�
=���HC�y�                                    BxY�m<  �          A33@��\�ٙ�?�33@��C���@��\��=q��  �ڏ\C��                                    BxY�{�  T          A
=@�p���ff?�  A��C�#�@�p���Q�aG���\)C�                                    BxY䊈  "          A\)@�Q��ۅ?�p�AffC�c�@�Q�����p�����
C�L�                                    BxY�.  "          A�@�����ff?�G�A%p�C��\@�����\�333��\)C�Q�                                    BxY��  	�          A�R@�z��ڏ\?��A(��C�R@�z���
=�!G���=qC���                                    BxY�z  
�          A33@�����
=?+�@��HC�^�@�����33����C��)                                    BxY��   
�          A33@��
��ff>u?��C��f@��
�ƸR��  �@Q�C�n                                    BxY���  
�          A�@���أ�>�?aG�C�aH@���Ϯ�����U��C���                                    BxY��l  
�          A�R@�z��Ӆ?�\)AMG�C�<)@�z����
�u�У�C��R                                    BxY��  �          A�
@�p���z�@=p�A���C��\@�p���Q�?��\@��
C���                                    BxY���  
Z          A�@������
@hQ�Aљ�C�.@�����ff?޸RAC\)C�=q                                    BxY�^  �          A(�@�\)��\)@#�
A��
C��@�\)��\)?�R@��C��                                    BxY�  "          A��@��H����?0��@��C��{@��H������Q�C�,�                                    BxY�+�  
�          Ap�@�(���{?\A'�C�}q@�(��˅��G��A�C�)                                    BxY�:P  �          A��@�\)��
=?�(�AZffC�Ff@�\)����=u>���C��=                                    BxY�H�  �          A��@����(�?�z�AS�
C���@����<�>k�C��                                    BxY�W�  "          A��@�ff��G�@��A��\C���@�ff��  ?�@k�C��\                                    BxY�fB  T          A  @�����@=qA���C�~�@�����(�?   @`  C�t{                                    BxY�t�  
�          @��@�p���{@\)A�\)C�*=@�p���@Q�A���C�o\                                    BxY僎  �          A�\@������H@e�A�(�C�H@�����?�\)AT  C��\                                    BxY�4  �          A=q@�  ��\)@W
=A���C��@�  ��Q�?ٙ�A@��C��
                                    BxY��  "          @�\)@�  �j=q@u�A�
=C���@�  ��z�@"�\A��C�=q                                    BxY寀  �          @�33@���S33@l(�A���C�t{@����Q�@!�A�G�C���                                    BxY�&  
�          @���@�33�7
=@vffBz�C�^�@�33�x��@3�
A�p�C�Y�                                    BxY���  �          @�
=@�=q���@�  B0
=C�P�@�=q�S�
@�ffB�
C���                                    BxY��r  "          @�@�����
=@�G�B(  C��@����B�\@�=qB
\)C�xR                                    BxY��  T          @���@�z���@�\)B 33C��=@�z��?\)@���B��C��H                                    BxY���  �          @��@������
@��B&�
C�=q@����*�H@�  Bz�C��q                                    BxY�d  "          @���@��H��ff@�B$�\C���@��H�L��@��B  C���                                    BxY�
  �          @��@�ff�<��@��RB��C�˅@�ff��=q@H��A�  C��)                                    BxY�$�  T          @�G�@���7�@��
B%33C�j=@����{@r�\A�G�C�B�                                    BxY�3V  
�          @�\@�Q�?���@��B?  A��@�Q�>�  @�  BOQ�@HQ�                                    BxY�A�  T          @�G�@���@\(�@�  B1{B\)@���?��H@�G�BUffA�33                                    BxY�P�  �          @��
@}p�@I��@���B>{B  @}p�?�{@ƸRB`{A�                                    BxY�_H  
�          @��@tz�@u�@�\)B*��B4ff@tz�@�@���BU=qA�p�                                    BxY�m�  T          A{@�  ��ff@	��Am��C���@�  �\>�Q�@(�C��3                                    BxY�|�  \          A�@�  ��=q@��A���C��R@�  ����?!G�@��C��
                                    BxY�:  �          A{@�\)��z�@5A��RC�
@�\)��\)?��@�(�C���                                    BxY��  T          A\)@����ff?Q�@��\C�'�@�����Ϳ���� ��C�AH                                    BxY樆  �          A=q@����R@�Ak�
C��)@����\>���@7
=C���                                    BxY�,  �          A�\@�p����@W�A�=qC��f@�p����R?�AZ�RC�S3                                    BxY���  T          A��@�����R@x��A��
C�n@����p�@�RA���C���                                    BxY��x  T          A�@ʏ\�mp�@���A�C���@ʏ\��\)@0  A�p�C��                                     BxY��  
�          Aff@θR�W�@�z�A�RC�=q@θR��@>{A��
C��                                    BxY���  �          A�R@�ff�E@���A�z�C���@�ff��(�@;�A�p�C�*=                                    BxY� j  �          A33@��R�y��@|(�A�C�h�@��R��(�@'�A�ffC�XR                                    BxY�  T          Aff@�ff�4z�@�33A�Q�C��f@�ff�y��@EA�33C��                                    BxY��  T          A�@�
=�2�\@s�
A�\)C�W
@�
=�q�@5�A��C��R                                    BxY�,\  
�          Aff@ȣ��+�@|��A��C�n@ȣ��mp�@@  A��RC���                                    BxY�;  "          A  @�z��Dz�@B�\A�p�C�H@�z��tz�@ ��Ae�C��H                                    BxY�I�  
�          A�@�  �   @�33B3��C��{@�  �b�\@���B{C�                                      BxY�XN  
�          A�@У׿��@��
B{C�,�@У��(Q�@���A��C�                                    BxY�f�  �          A ��@љ��ff@�G�A��C�!H@љ��Z�H@K�A�z�C�4{                                    BxY�u�  �          A{@���W�@G�Ag�
C�AH@���s�
?u@�=qC���                                    BxY�@  "          A  @��W�?�A�C���@��h��>Ǯ@,��C��R                                    BxY��  "          A  @ᙚ�g�@\)A|(�C�e@ᙚ���
?��@�z�C���                                    BxY硌  T          Ap�@��n�R?E�@��C���@��s33��{��C�Z�                                    BxY�2  
�          A33@�33���
�����G�C�]q@�33�w
=��(�� ��C�&f                                    BxY��  �          A@ۅ��33����F�HC�˅@ۅ�j=q�>{���C���                                    BxY��~  
Z          A��@У����׿����\��C��H@У��qG��J�H��ffC���                                    BxY��$  
�          Aff@�{���@��
B=qC�q@�{�'
=@�G�A�G�C��R                                    BxY���  T          A ��@��H�˅@���B$=qC�aH@��H�A�@��HB�RC�Ff                                    BxY��p  �          A(�@�
=��G�@��HB4��C���@�
=�(��@��B p�C��q                                    BxY�  T          A@���5@�33BZ��C��f@���#�
@�(�BE
=C�|)                                    BxY��  �          @�{@!녿�ff@�33B�\C�K�@!��e�@ӅBa��C��                                    BxY�%b  �          A ��@U���  @�
=B��{C���@U��@��@�z�B_C��{                                    BxY�4  �          @��R@�׿@  @���B���C�Ǯ@���7
=@�  By�HC�/\                                    BxY�B�  �          @�z�>�����z�@�B��C�q�>����\)@�Q�Bm�C�#�                                    BxY�QT  �          @��@e��=q@���Bq��C�R@e�^{@�p�BL(�C��
                                    BxY�_�  
�          A ��@P  ��
@��Bu{C��@P  �}p�@���BH�HC��\                                    BxY�n�  �          @���@"�\���@��B���C�l�@"�\�a�@ҏ\Bb
=C���                                    BxY�}F  T          @��R@���{@��B�33C���@��hQ�@��Bd33C�O\                                    BxY��  
�          @�=q?�Q���@陚B�(�C�H�?�Q����H@��B]p�C�@                                     BxY蚒  T          @��
@\)���@���BdffC�]q@\)�J�H@�(�BDQ�C�                                    BxY�8  �          AG�@�ff��@�
=B+�
C��@�ff�9��@��B�
C���                                    BxY��  
�          A�
@�z��\)@��HB��C�'�@�z��5@}p�A�C�n                                    BxY�Ƅ  �          A@�\)��@��A�C�E@�\)�%�@r�\A�  C��                                     BxY��*  �          A��@�\�h��@x��A�(�C�q�@�\���H@^�RA��C���                                    BxY���  T          A�@�
=����@�\)A�G�C��@�
=�=p�@c33A�
=C��H                                    BxY��v  
�          A��@��H�8Q�@�\)B�C���@��H��Q�@`  A�{C��{                                    BxY�  "          A(�@�Q��!�@�p�BffC�aH@�Q��j=q@c33A�p�C�U�                                    BxY��  "          A��@Ӆ���@�=qB
=C�\@Ӆ�*�H@�  A�Q�C�H                                    BxY�h  �          A=q@�ff��Q�@�=qB�HC���@�ff�HQ�@vffA�=qC�|)                                    BxY�-  �          A
=@��Ϳz�H@�ffB�
C���@�����R@��A��C��                                    BxY�;�  "          A��@�(���G�@�
=B33C�{@�(����@���B�\C�l�                                    BxY�JZ  
�          A��@Ǯ?��@�=qB
=A]@Ǯ=�Q�@�G�B!G�?Q�                                    BxY�Y   �          A��@���@W�@�ffB�\A�\)@���@G�@�ffB{A�33                                    BxY�g�  �          AQ�@�  @��@�  A߮B ��@�  @e�@�Bp�A�G�                                    BxY�vL  �          A��@�Q�@c33@mp�A�=qA�ff@�Q�@��@�G�BffA��
                                    BxY��  �          @�p�@j�H�Dz�?���A���C�f@j�H�_\)?��A:{C�8R                                    BxY铘  "          @����
=��=q��ff��p�C�w
��
=���H����\)C�W
                                    BxY�>  �          @�R������Ϳ�R��  C��׿����ff�&ff����C�                                    BxY��  T          @�ff?�ff��\�z����HC�*=?�ff�����R��\)C���                                    BxY鿊  �          A33@�  ��\)?�AL  C�^�@�  ��\)=u>ǮC��\                                    BxY��0  "          A�H@J�H���?���A�
C���@J�H��  �333��p�C�p�                                    BxY���  �          A  @{���?G�@��C���@{��\)������C�ٚ                                    BxY��|  T          A  @�R��>aG�?\C��q@�R��{���W\)C�
                                    BxY��"  
�          A��?�p��{?@  @�{C�#�?�p�� �׿��H�!C�33                                    BxY��  �          Aff?��R�(�?333@�G�C�G�?��R�ff����(��C�W
                                    BxY�n  T          A�?�\��H?0��@��C���?�\��ÿ˅�,��C��                                     BxY�&  �          AQ�?c�
�
=?!G�@���C�?c�
��ÿ���1p�C��                                    BxY�4�  �          A�?����(��k���=qC�S3?���������H���HC���                                    BxY�C`  �          A�H@G�� �ÿs33�ϮC���@G�����C33��{C�c�                                    BxY�R  
�          Ap�@=q���H��=q�O\)C���@=q�ڏ\�qG���(�C��{                                    BxY�`�  T          A�?�33� zΌ��� (�C��=?�33���
�aG����
C�L�                                    BxY�oR  �          Aff?�{��z��p��uG�C���?�{�������R��{C�q�                                    BxY�}�  
�          A?����{��  �%��C��R?�����ff�e��˅C�0�                                    BxYꌞ  }          A��?�
=�ff���}p�C���?�
=���R�,(���p�C�<)                                    BxY�D  T          A=q?У���p��   �^{C���?У����
��  ��C�s3                                    BxY��  "          A{?�\)�����	���n�RC��\?�\)�����z����C�y�                                    BxY긐  �          A=q?˅������H��C��q?˅��  �u��ڏ\C�,�                                    BxY��6  �          Ap�@33���R�^�R���C�@33��R�:=q���C���                                    BxY���  T          AG�@{����������C��)@{��G��HQ����HC�\)                                    BxY��  �          AG�@|����  ��\)��(�C�G�@|�����	���qp�C��{                                    BxY��(  �          AG�@g���
=����qG�C���@g���\�p���p�C���                                    BxY��  "          A�R@��H��녿���p  C���@��H���=q����C�K�                                    BxY�t  �          A{@>{��녾����.{C�g�@>{��ff������HC��q                                    BxY�  T          A��@#�
�����u����C��@#�
�陚�;���C��\                                    BxY�-�  �          A�@<��������R��
=C�\)@<����(��%��Q�C���                                    BxY�<f  �          Aff@5����B�\��
=C��\@5�����/\)���RC�~�                                    BxY�K  �          Az�@C�
��Q�\�)C�
@C�
��(��XQ����C�                                      BxY�Y�  "          A{@1����H��
�k
=C�]q@1��ҏ\�vff���HC�s3                                    BxY�hX  �          A�\@c�
��33�޸R�K\)C���@c�
���Z�H�̸RC���                                    BxY�v�  �          A ��@���6ff����0{C�!H@����{��33�FC�g�                                    BxY녤  T          @�(�@�����Q���� Q�C���@����r�\�����)�C���                                    BxY�J  �          A�\@�{�tz���G��!  C�"�@�{���\�?�HC�/\                                    BxY��  �          @��R@���X����G���C��@���G���\)�9�HC�q                                    BxY뱖  �          @��R@�G�������{���C�|)@�G���G����H�IC�,�                                    BxY��<  �          A�@����{?(�@�=qC�(�@������fff��(�C�>�                                    BxY���  �          A�@�  ��z�>�@O\)C�G�@�  ��녿�=q���RC�w
                                    BxY�݈  �          A33@���ff�#�
���C��3@���
=�ٙ��AC�}q                                    BxY��.  �          @��@W�������p���C���@W��y������<��C�o\                                    BxY���  "          @�p�@�\)����������C�B�@�\)�AG���=q�<
=C�=q                                    BxY�	z  �          @�\)@����(���Q����C�K�@���`  �����$�RC�C�                                    BxY�   T          @�{@�����
��  ��C���@���P����ff�!
=C�3                                    BxY�&�  �          @��@��������p��C��@���=p����\�C  C�Ff                                    BxY�5l  �          A(�@�Q��
=��(��6G�C�(�@�Q쾀  ���
�@p�C���                                    BxY�D  T          A
=@��?�33�ə��K��A���@��@XQ������0p�B�
                                    BxY�R�  �          A{@p  @r�\�\�B�HB533@p  @����
=���BW��                                    BxY�a^  �          @�(�@c�
@E����N(�B#=q@c�
@�p���Q��%�\BL�                                    BxY�p  �          @��\@�33������S\)C��
@�33?�Q���z��J�A�G�                                    BxY�~�  �          @��
@9��?�R����{A@��@9��@G��љ��r�
B(�                                    BxY�P  �          @�\)?�\)@��љ��RB^
=?�\)@w���G��XG�B�u�                                    BxY��  �          @�R�33��33�H����Q�Cy�f�33����z��"�RCv^�                                    BxY쪜  �          @��R@�\)@7
=��{�6��B
=@�\)@�=q��33���B){                                    BxY�B  T          @�ff@B�\@�����p��2(�BZ��@B�\@�z���  �p�Brp�                                    BxY���  �          @��H@%�@�  ��\)�{Bz��@%�@�p��J=q�У�B��
                                    BxY�֎  �          @�Q�@#33>���Q�{A%�@#33?�Q���
=�wQ�B�H                                    BxY��4  T          @��
@&ff?�G���Q��}z�Bp�@&ff@L(���p��X
=BKQ�                                    BxY���  T          @�33@vff?L�����j33A:�R@vff@�\����V(�A��
                                    BxY��  �          @��R@�������G��hC�3@���?�z���(��`\)A�\)                                    BxY�&  �          @�p�@q�?�����
�o�
A�@q�@���ə��]�\A��                                    BxY��  T          @�(�@�Q�?�Q���z��TG�Ay�@�Q�@!���ff�>A��                                    BxY�.r  �          @ᙚ@W
=?�Q�����h�RA��\@W
=@   �����N�B��                                    BxY�=  �          @�(�@tz�@z�����Q�A㙚@tz�@Tz���\)�2=qB#�H                                    BxY�K�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxY�Zd              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxY�i
              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxY�w�  
�          @��
@�\)>u��(��&��@   @�\)?�����ff�(�A^{                                   BxY�V  �          @�p�@�>�33��\)�
=@^�R@�?�������A`z�                                   BxY��  �          @�@��>�{�����!�@^{@��?�
=��33���Ab�H                                   BxY���  �          @�R@�����\�����4��C��f@����Q����R�;��C�t{                                   BxY��H  T          @�G�@�  �����\�4�C��@�  �����
=�Fz�C��3                                    BxY���  �          @���@���?�(���p��  A`(�@���@��~{���A�G�                                    BxY�ϔ  "          A�H@�\)�\)��(���C���@�\)?����(����@�33                                    BxY��:  T          A{@�=q�������\)C�.@�=q�L����ff���C���                                    BxY���  �          @��R@�33��R��\)� ��C�U�@�33��(����
�=qC���                                    BxY���  �          @�=q@����(�����.G�C�R@��������8��C�l�                                    BxY�
,  T          @��@���\(���p��Kp�C��)@��>����R�Mz�@�Q�                                    BxY��  �          @�
=?�z��AG���{�k�HC�%?�z�У��θR�\C���                                    BxY�'x  ]          @�33?�G�������  �W�C��)?�G��)����  aHC�                                    BxY�6  K          @�ff@33��p�����9�
C��)@33�W���Q��g�C���                                    BxY�D�  �          @�?޸R��(�����)�
C��\?޸R�n{��ff�X��C��f                                    BxY�Sj  T          @�(���\�]p���  �v��C��ÿ�\���H��338RCz�                                    BxY�b  �          @�33�����������Ct޸��녿c�
����{CZ{                                    BxY�p�  �          @��>��H��33�����S(�C�=q>��H�@  �ҏ\u�C��\                                    BxY�\  �          @�(�?   ������{�)33C���?   ���R����Z��C�c�                                    BxY�  �          @�\?�=q������\)�"�C�\)?�=q��(���Q��Q�C��                                     BxY  �          @�p�@Q����
���
���C�C�@Q��E������C�C�c�                                    BxY�N  T          @ᙚ@:=q��=q��{��C��{@:=q�s�
���
�;�\C��{                                    BxY��  �          @�z�@!G�����Z=q��G�C�h�@!G����
������C��                                    BxY�Ț  T          @�=q?�p���(��<(���  C�O\?�p���33�����
�HC�XR                                    BxY��@  �          @�G�>�p����
�aG���=qC�s3>�p��������ffC��                                    BxY���  
�          @�z�?�����33�
=���\C�.?�������z����C�xR                                    BxY��  �          @߮@
=�����l(��\)C�n@
=�j�H����<��C��\                                    BxY�2  
�          @�=q>�33@z���ff�B�>�33@QG���(��j33B�                                    BxY��  �          @�(��G�@�
=���
�8�B�=�G�@�{��=q�(�Bܣ�                                    BxY� ~  "          @�녿p��@n{����bffB�B��p��@�33����3ffB��
                                    BxY�/$  �          @ʏ\���@Q�����{�B�\���@W���ff�N33B��)                                    BxY�=�  �          @�ff���@B�\���\�j�B����@��\���H�=33B�(�                                    BxY�Lp  �          @\�!G�?�z���{��B�k��!G�@C33��p��k�
B�aH                                    BxY�[  �          @���?���p����  �qC��?��=�G����\�
@�{                                    BxY�i�  �          @�  @1G���z��<(���(�C�s3@1G��|(��tz��C��
                                    BxY�xb  �          @ə�@*�H�����������C���@*�H�����Y���
=C�9�                                    BxY�  �          @�{@1���(��u�.{C��3@1����ÿfff�&{C�E                                    BxY  �          @�=q?ٙ��[�@=p�B{C�5�?ٙ��|��@p�A�G�C���                                    BxY�T  
�          @�Q�@��H��\)?�{A((�C�XR@��H��z�>8Q�?�\)C��R                                    BxY��  �          @�@��\��
=?�A[33C���@��\��ff?�@|��C�%                                    BxY���  T          @���@�33��(�?��HAiC��\@�33����?0��@��
C�)                                    BxY��F  �          @�G�@�ff��33?�=qA�HC��)@�ff�Ϯ=#�
>���C�t{                                    BxY���  T          @��@��
��z�?�{A�C�xR@��
�Ǯ���xQ�C�AH                                    BxY��  �          @���@������?�
=Ao\)C�� @�����ff?k�@ᙚC���                                    BxY��8  �          @��\@�{���R?�  A2=qC�q@�{��p�?
=@�z�C�}q                                    BxY�
�  �          @���@��H��(�?�A(z�C�@��H��=q>�(�@K�C���                                    BxY��  
�          @���@�����33?�\AS�C��@������H?#�
@���C�j=                                    BxY�(*  �          @�33@��
����?\(�@�=qC�g�@��
�Å�����p�C�H�                                    BxY�6�  �          @�
=@������@W�A��C���@�����ff@
=qA{33C�aH                                    BxY�Ev  T          A�@�
=��p�@Q�A���C���@�
=��{@z�Al��C�aH                                    BxY�T  
�          A{@�z���(�@uA�(�C���@�z���Q�@+�A��RC��                                    BxY�b�  �          A�@{���z�@�
=A�z�C�o\@{����H@C�
A�z�C��                                    BxY�qh  �          Az�@x����ff@s33AۅC�
=@x�����@#33A��C���                                    BxY��  �          A�@q��ə�@{�A�33C�w
@q���{@*=qA�
=C�H�                                    BxY���  
�          A(�@c�
�θR@mp�AծC�t{@c�
��G�@�HA�z�C�l�                                    BxY�Z  �          A=q@h�����H@l(�AиRC�xR@h����p�@Q�A��
C�y�                                    BxY�   
Z          A\)@���\@c�
A��HC��@����z�@ffA�(�C��                                    BxY�  "          A ��@�
=���
@�A���C��@�
=���@FffA�=qC�p�                                    BxY��L  T          AG�@�{��  @���BG�C��=@�{��Q�@`��A�Q�C�Ф                                    BxY���  "          A  @p������@���A�=qC��{@p����p�@5A���C��\                                    BxY��  
�          A  @K���
=@c�
Ạ�C���@K����@��A~=qC��                                    BxY��>  T          A\)@|(�����@~�RA�z�C��q@|(�����@5�A��HC���                                    BxY��  T          A z�@�
=���R@>{A��C�%@�
=���?�A_�
C�                                    BxY��  �          @�?��
���@3�
A���C��q?��
���?��HA*ffC���                                    BxY�!0  T          Ap�@A��ᙚ@\)A�C���@A���(�?�AG�C�(�                                    BxY�/�  "          AQ�@\(���?�AW�C���@\(���
=?z�@�G�C�Z�                                    BxY�>|  T          A ��@ ����33@�ffB!�C�{@ ����{@��A�C��f                                    BxY�M"  T          A (�@.{��@�\)B Q�C��)@.{���H@C�
A�z�C��{                                    BxY�[�  T          @��@�
��ff@i��A�  C���@�
��@��A�  C��
                                    BxY�jn  �          A@7���z�@c33A���C��@7����@�
A�\)C���                                    BxY�y  �          A�\@aG����@U�A���C�  @aG���G�@�Ap��C�J=                                    BxY�  T          A=q@A����@~{A��
C���@A���  @1�A�  C��
                                    BxY�`  T          A (�@Q���=q@��A�C�33@Q���{@;�A��C��                                    BxY�  T          A   @G���z�@c33A�ffC��@G����@��A��C�&f                                    BxY�  �          AG�@�����\)@{A���C�n@������?�ffA��C��                                    BxY��R  �          A  @��
��33?�z�A��C���@��
�׮>�  ?�z�C��)                                    BxY���  �          A
=@����z�?�33AP(�C��@���ۅ?:�H@�
=C���                                    BxY�ߞ  �          AG�@����ʏ\@"�\A�  C�� @�����p�?�A��C�Ǯ                                    BxY��D  �          A{@�{��p�@'�A��C��@�{�أ�?�p�A#�C�Y�                                    BxY���  T          Aff@��H����@.�RA�C��3@��H��Q�?ǮA+�C��H                                    BxY��  �          A�\@�����
=@EA�ffC��@�����z�?���AV�HC�C�                                    BxY�6  �          A�\@�  ����@J=qA���C��f@�  �޸R@G�A^=qC���                                    BxY�(�  
�          A
=@����\)@[�A��\C�/\@����
=@33A}C�E                                    BxY�7�  �          A\)@~�R���\@��A�C�*=@~�R��@E�A�p�C��H                                    BxY�F(  �          A{@�����z�@�G�A�RC���@�����
=@>{A�{C�w
                                    BxY�T�  T          A�H@j�H��=q@��A�(�C��\@j�H���@@  A�ffC�k�                                    BxY�ct  T          A{@a����H@N�RA�{C���@a����@�
AeG�C���                                    BxY�r              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxY��              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxY�f              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxY�  
�          Ap�@h������@�G�A��C�Q�@h����Q�@P  A�p�C�(�                                   BxY�  �          A��@A���33@�B�\C���@A��У�@k�A�
=C�u�                                   BxY�X  
�          @���?����p�@�z�B&�HC�K�?����p�@�B�
C���                                   BxY���  �          @��?�Q���
=@��\BffC��?�Q��ڏ\@R�\A���C��                                   BxY�ؤ  �          A ��@33��G�@fffAՅC�\)@33���@\)A��\C��f                                    BxY��J  �          A@(���\@��A�  C�1�@(���?�(�A
�HC��                                     BxY���  �          @�@E��ff@�\Ap��C���@E��?n{@�=qC��                                     BxY��  �          A ��@(����>�G�@J�HC��{@(����
=�:�H���C��)                                    BxY�<  T          A�@o\)��
=�Vff��=qC���@o\)��(�������C��\                                    BxY�!�  �          @�{@9����{��
=�C��@9�����������0�RC�T{                                    BxY�0�  T          @�@~�R��33�$z����C��@~�R��z��[���
=C�7
                                    BxY�?.  
�          A�@����\)?��RAaC�'�@����
=?��A   C��=                                    BxY�M�  �          Aff@�(�����?�33AZffC���@�(����
?��\@陚C�@                                     BxY�\z  T          A�@�����@%A�z�C���@����=q?�\AIp�C���                                    BxY�k   �          A�@�
=��=q@�AmG�C���@�
=���?�33A�C�R                                    BxY�y�  �          Aff@������@�An�\C���@����Ǯ?�A��C�k�                                    BxY�l  �          @���@Ϯ�Fff@-p�A�33C�>�@Ϯ�]p�@{A�z�C��{                                    BxY�  �          @�(�@љ��K�@.�RA�z�C�@љ��c33@�RA���C���                                    BxY�  �          A ��@�{����?���A3�C���@�{���\?W
=@��RC�S3                                    BxY�^  
�          A ��@�������?��HAd(�C��@�����  ?�  A(�C�7
                                    BxY��  {          Az�@У���ff?��AIG�C�)@У����?���@�z�C���                                    BxY�Ѫ  �          A�@�z���p�@#33A�33C��\@�z����?��AUp�C��)                                    BxY��P  T          A Q�@��R��\)@EA��C��\@��R���
@Q�A��C�l�                                    BxY���  "          @��@��R���@���A��C�.@��R��33@VffAǮC���                                    BxY���  	�          A   @�
=����@,��A�G�C�N@�
=��\)?�z�Aa��C�xR                                    BxY�B  
e          A ��@�  ��z�?�z�A"=qC��
@�  ��G�?.{@��HC�P�                                    BxY��  I          A ��@�=q��\)@6ffA�z�C�u�@�=q���\@
=qAz{C�q�                                    BxY�)�  "          A Q�@������@Tz�A�ffC�*=@������@,��A��C��q                                    BxY�84  �          @�
=@�z����\@:�HA�\)C�q�@�z���{@��A�Q�C�aH                                    BxY�F�  "          @��@��
��
=?�\)A!�C�=q@��
���?��@�ffC���                                    BxY�U�  �          A=q@����p�=�?Y��C��
@����z�(����(�C��\                                    BxY�d&  �          A�\@����\���z�\C�  @���\)�:�H��G�C��                                    BxY�r�  �          @���@�33��G�����  C�{@�33���?\)��z�C�1�                                    BxY�r  T          @��@>�R�z=q���DffC��\@>�R�G
=�����[�
C��f                                    BxY��  "          @�z�@W��~{��
=�4(�C�%@W��O\)��ff�J�\C�
=                                    BxY���  "          @��@�=q����$z���C��R@�=q�~�R�HQ���G�C��3                                    BxY��d  "          @�ff@�ff�����G��C��=@�ff���Ϳ��_�
C�B�                                    BxY��
  �          @�(�@�  �����\)�D(�C�˅@�  ��\)�
=��  C�s3                                    BxY�ʰ  T          @�\)@������H��  ��C�@ @�����(����H�t  C��=                                    BxY��V  
�          @�
=@��\��p���{�H��C��@��\��p���R��{C�b�                                    BxY���  �          @���@�
=�������c�C���@�
=�y�������HC�g�                                    BxY���  "          @��@�{�Z=q�8Q���\)C�z�@�{�R�\��33�Q�C���                                    BxY�H  T          @��@�p��C�
�\)��  C�q�@�p��AG����uC��
                                    BxY��  "          @�{@ٙ��[�=L��>�33C���@ٙ��Z=q�Ǯ�:=qC���                                    BxY�"�  "          @��@����u���33�
{C���@����j=q��\)�C\)C�`                                     BxY�1:  �          @�(�@ٙ��J�H�������C�z�@ٙ��@�׿�p��4Q�C�
=                                    BxY�?�  �          @�33@޸R�%��\)��z�C��{@޸R�  �2�\���C��                                    BxY�N�  
�          @�p�@陚�333������C��H@陚�(Q���
�2�HC�
                                    BxY�],  �          @�Q�@����[���G��S�C�^�@����L(��
=q����C�33                                    BxY�k�  
Z          @�  @��H�����\��z�C���@��H��  �:�H��{C��                                    BxY�zx  �          @�@\)�\�@����  C���@\)���n�R���
C��\                                    BxY��  �          @�Q�@W
=�����G���G�C���@W
=��G�����
C��R                                    BxY���  �          @�
=@�  �i����(���C�<)@�  �Dz������&p�C���                                    BxY��j  
�          @��@�����'����C��{@��Ϳ�G��6ff���\C��{                                    BxY��  �          @�=q@��H��z�������C���@��H����5��Q�C���                                    BxY�ö  �          @��
@�G��G���\)��C�}q@�G��   �����p�C���                                    BxY��\  
�          @�p�@��K����
�C���@��@�׿���@(�C���                                    BxY��  T          @���@��H�+�����(�C��)@��H�!녿�
=�'�C��                                     BxY��  T          @��@�  �33�������\C�^�@�  ��zῦff�  C�ٚ                                    BxY��N  
�          @�ff@�ff�\)�.{��p�C�@�ff��;�ff�Q�C��H                                    BxY��  T          @�@��H�p  �G�����C�,�@��H�h�ÿ������C���                                    BxY��  
�          @��R@�{�z�H�#�
���
C�XR@�{�x�þ�ff�Z=qC�l�                                    BxY�*@  "          @�z�@�G��p�?��
A3�C���@�G��'
=?��RA{C�&f                                    BxY�8�  T          @�z�@���dz�>�33@ ��C��f@���e�L�;�Q�C�xR                                    BxY�G�  �          @���@�
=��ff�(���Q�C�}q@�
=����������C��f                                    BxY�V2  
�          @��@��\���
�Q���C��f@��\���H�8�����HC���                                    BxY�d�  �          @�p�@������������C��@��������2�\���C��=                                    BxY�s~  �          @�@��\���Ϳ��H�O�
C��)@��\��{�  ���C���                                    BxY��$  T          @�z�@�������5���HC��@�����  ��=q� ��C���                                    BxY���  �          @�@�ff�Å�G���p�C��@�ff������,��C�J=                                    BxY��p  
�          @�\@�����������C�\)@����  �.{����C�|)                                    BxY��  
�          @��H@w
=��{������C�u�@w
=���
�xQ���C��
                                    BxY���  �          @�ff@l(���ff>W
=?�=qC��q@l(���{���c33C���                                    BxY��b  T          @��H@�ff�o\)?���AI�C��\@�ff�xQ�?�
=A33C�s3                                    BxY��  �          @��@�=q��
=>aG�?�33C�z�@�=q��
=����G�C�|)                                    BxY��  T          @���@�ff���H?.{@�\)C�"�@�ff��(�>\)?�=qC�f                                    BxY��T  "          @�Q�@�
=����?\(�@��
C�J=@�
=���>���@!G�C�!H                                    BxY��  
�          @�  @�p��������H�qG�C�l�@�p���
=����33C��q                                    BxY��  �          @��R@�����=q�8Q���33C���@�����
=��G��
=C��
                                    BxY�#F  �          @���@�������>#�
?��HC�
@�����G���Q��*�HC�q                                    BxY�1�  �          @�@\����?˅AD��C�<)@\��p�?�Q�A33C�˅                                    BxY�@�  �          @�Q�@���l(�@P  A�{C�0�@���~�R@8Q�A�=qC�0�                                    BxY�O8  T          @�33@���j=q@�HA�\)C�%@���w�@�
AvffC�o\                                    BxY�]�  �          A��@���?�33@�z�B{AMG�@���?�  @�  B	=qAQ�                                    BxY�l�  
�          A  @陚�B�\@Dz�A�{C�AH@陚��@A�A�{C�"�                                    BxY�{*  T          AG�@��
��@�A�33C�\@��
�B�\@�A�C�@                                     BxY���  "          A�@��&ff@>{A�(�C���@��k�@9��A�C��\                                    BxY��v              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxY��              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxY���              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxY��h  	�          A=q@��aG�?�(�AYC�7
@��k�?�33A5C��{                                   BxY��  "          Ap�@ۅ��{?��@�(�C��@ۅ��\)>8Q�?�  C��{                                   BxY��  
�          A\)@�
=���>���@�C�!H@�
=��=q�L�;�p�C�R                                   BxY��Z  �          A�R@׮����?�{A�C�q�@׮��  ?z�H@��
C�%                                   BxY��   
�          A  @����
=?.{@�  C�z�@������>��R@(�C�Y�                                    BxY��  T          A(�@��n{?�{A�
C�ff@��tz�?��@�
=C�{                                    BxY�L  �          A�@��^�R?�Q�AZ{C�\@��h��?У�A7�C��3                                    BxY�*�  �          A�\@�p��[�?�{ARffC�7
@�p��dz�?ǮA0z�C��H                                    BxY�9�  T          A  @������@��Ar{C��\@����ff?�ffAJ�\C�.                                    BxY�H>  
�          A�
@��H����?�G�@��HC�33@��H��33?.{@��C��q                                    BxY�V�  �          A�H@��
���?�Q�AffC��3@��
��=q?L��@�
=C�xR                                    BxY�e�  "          A��@����z�?��RAb�HC���@����G�?�=qA4(�C�`                                     BxY�t0  �          A��@�\)��=q?�(�A@z�C���@�\)��ff?�=qA(�C�P�                                    BxY���  �          A��@�������?�{A33C�+�@������?z�H@أ�C��                                    BxY��|  �          A��@�=q��33?�G�A(��C���@�=q��ff?���@�C�E                                    BxY��"  �          A��@�z�����?���@��RC��@�z���33?L��@��\C��H                                    BxY���  �          A=q@�p����
?��HA��C�@�p���ff?c�
@��
C��\                                    BxY��n  �          Aff@�G���=q?�R@��C�e@�G����>��R@C�J=                                    BxY��  T          Aff@�  �g
=?��@�z�C�(�@�  �k�?E�@��C���                                    BxY�ں  T          A�@��H��p��aG���(�C�h�@��H���H��G����C��H                                    BxY��`  �          A�@�����p��\�'�C�Q�@�����G����T��C���                                    BxY��  �          Az�@�p����H�Ǯ�*{C��@�p����R��(��W33C�>�                                    BxY��  T          AQ�@ҏ\���R?xQ�@��
C��{@ҏ\��Q�?
=@�Q�C���                                    BxY�R  
�          A�H@ָR���\�����$z�C�� @ָR���ÿQ���G�C��)                                    BxY�#�  �          A��@ƸR�\�0�����C�Ф@ƸR����L(����C�S3                                    BxY�2�  "          A(�@�33�����J�H��
=C��@�33����e��(�C��
                                    BxY�AD  
�          A��@��R��Q쿬���(�C���@��R���Ϳ�  �@��C�E                                    BxY�O�  
�          A��>�����
=���R�33C���>���������(���C���                                    BxY�^�  �          A
=<#�
���^{��(�C��<#�
��33�|����C��                                    BxY�m6  T          A
=@:=q��녿�  �2=qC�0�@:=q��{��p��l  C�\)                                    BxY�{�  "          A�
?�\)���׿���0(�C�� ?�\)������
�l��C��
                                    BxY���  "          A�@�(����?�\)A{�C���@�(���33?��
AMG�C�Q�                                    BxY��(  T          A�@)����Q��.�R��(�C�Z�@)������L(���ffC��                                     BxY���  
�          A�@�����@���B	��C�@����33@��RA���C�/\                                    BxY��t  �          A\)@s33����@�B��C���@s33���H@��\Bp�C�R                                    BxY��  �          Az�@�  ���\@|��A噚C��@�  ��=q@dz�A�C�z�                                    BxY���  T          A(�@�������?�p�AEp�C�=q@�����(�?���A�C��                                    BxY��f  "          A�\@p����Q�?0��@�
=C��@p����G�>�  ?�C�                                    BxY��  T          A=q@�{���H?���A0��C��@�{��?���A��C�t{                                    BxY���  "          A(�@��
��\)?���A�C�E@��
�ᙚ?k�@θRC�%                                    BxY�X  �          A{@�{���\?���@��HC�
@�{��z�?O\)@��C��                                    BxY��  
�          A�\@�ff��=q?��A�C��@�ff����?���@��C���                                    BxY�+�  "          A�@�(���  ?�Q�A$Q�C���@�(���=q?���A��C��                                    BxY�:J  �          A@�{����?���A�C��@�{���H?��RA�C��                                    BxY�H�  "          Aff@�(��!G�?^�R@���C�#�@�(��#�
?:�H@�Q�C��q                                    BxY�W�  
�          A�Aff?#�
?!G�@�@�\)Aff?��?+�@�p�@�
=                                    BxY�f<  T          A�H@���?�\)�k�����AT��@���?�׽��Y��AU�                                    BxY�t�  T          AG�@�(�=�@N{A�33?fff@�(��u@N{A�G�C��f                                    BxY���  �          Aff@��@�?�33A Q�Al(�@��@G�?��\A��Ad��                                    BxY��.  �          A��@�(�@�ff?aG�@��B�@�(�@���?���@�33B\)                                    BxY���  T          A@���?�\)?��RA^ffAW\)@���?�G�@�Ai��AK33                                    BxY��z  "          A=q@陚@A�@0  A�G�A��\@陚@8Q�@9��A�z�A�z�                                    BxY��   T          A��@�ff@?�ffA�
A�(�@�ff@G�?�A*�\A�{                                    BxY���  �          A(�A=q�xQ�@  ��\)C��RA=q�n{�L�����\C��)                                    BxY��l  �          A�@�z�#�
?�p�Aap�C�h�@�zᾅ�?�(�A`(�C��                                    BxY��  �          A��@�
=@_\)?�@�=qA�Q�@�
=@]p�?@  @��\Ạ�                                    BxY���  �          Ap�A (�@
�H?#�
@�{As33A (�@��?@  @�{Ao\)                                    BxY�^  "          A
=A\)?��ÿG����A,��A\)?��Ϳ333����A0��                                    BxY�  �          A
=A�R���?&ff@�C�S3A�R��?�@xQ�C�9�                                    BxY�$�  |          A
ffAp�����Ǯ�(��C��{Ap���\���Mp�C�f                                    BxY�3P  "          A{A���z�����$��C�y�A���=q��33�,��C��R                                    BxY�A�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxY�P�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxY�_B              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxY�m�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxY�|�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxY��4              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxY���              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxY���              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxY��&              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxY���              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxY��r              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxY��              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxY��              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxY� d              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxY�
              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxY��              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxY�,V              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxY�:�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxY�I�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxY�XH              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxY�f�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxY�u�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxY��:              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxY���              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxY���              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxY��,              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxY���              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxY��x              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxY��              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxY���              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxY��j              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxY�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxY��              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxY�%\              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxY�4              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxY�B�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxY�QN              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxY�_�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxY�n�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxY�}@              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxY���              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxY���              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxY��2              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxY���              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxY��~              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxY��$              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxY���              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxY��p              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxY�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxY��              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxY�b              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxY�-              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxY�;�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxY�JT              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxY�X�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxY�g�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxY�vF              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxY���              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxY���              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxY��8              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxY���              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxY���              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxY��*              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxY���              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxY��v              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxY��              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxY��              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxY�h              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxY�&              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxY�4�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxY�CZ              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxY�R               ��O���O���O���O���O���O���O���O���O���O���O�                                  BxY�`�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxY�oL              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxY�}�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxY���              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxY��>              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxY���              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxY���              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxY��0              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxY���              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxY��|              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxY��"              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZ �  �          A�
�333@�
=��G��a33B�Ǯ�333@�  �أ��`p�Bų3                                    BxZ n  �          A33��Q�@�p�����W��B��f��Q�@�{�ٙ��W  B۽q                                    BxZ   T          A����
@��\��p��;33Bҏ\���
@�33�����:��B�z�                                    BxZ -�  T          A33���@���33�&\)B۳3���@�{���\�%�Bۣ�                                    BxZ <`  T          Aff���
@��H���
�P�\B�{���
@���˅�P(�B�
=                                    BxZ K  �          Aff�Y��@�ff��
=�G{Bƅ�Y��@�ff�ƸR�FBƀ                                     BxZ Y�  �          @�
=�L��@�33�ə��Q��B��׾L��@�33��G��QffB���                                    BxZ hR  �          @�ff�333?���  ¤�{C��333?��Ϯ¤�=C�
                                    BxZ v�  �          @�?�����������?G�C���?�����������?\)C��
                                    BxZ ��  �          @�\)?5��33��  �633C��\?5��33��  �6(�C��\                                    BxZ �D  �          @�?��Ϳ�z���\).C�R?��Ϳ�z���\)#�C��                                    BxZ ��  �          A�@<�����u��ffC�/\@<�����u���  C�.                                    BxZ ��  �          A��@"�\��ff�s�
��(�C��{@"�\��R�s33�ә�C��3                                    BxZ �6  �          @�\)@0  �����=q�7�\C��@0  ���
����7=qC�{                                    BxZ ��  �          A��@P  �6ff��Q��e��C�g�@P  �7
=��  �e\)C�W
                                    BxZ ݂  �          Az�@P  �$z����
�rz�C�ٚ@P  �%���r(�C��H                                    BxZ �(  �          A (�@Y���=q�أ��i��C�Q�@Y������Q��iG�C�8R                                    BxZ ��  T          @��?�z��L����\)�T�RC�u�?�z��Mp���
=�T�C�ff                                    BxZ	t  T          A	G�>������=q��G�C�y�>����p��Q���C�xR                                    BxZ  
�          A�H��(��녿B�\���RC�1��(��녿:�H��Q�C�1�                                    BxZ&�  
�          A\)��  ��
=�AG���Q�C�����  ��\)�?\)��z�C���                                    BxZ5f  �          A�R�O\)��=q�������C��3�O\)���H���
���C��{                                    BxZD  �          A	p��L����  �����!�
C�!H�L����������� C�"�                                    BxZR�  �          A  ?����  ������C��f?�����������\C���                                    BxZaX  T          AQ�?
=q��ff��G��"=qC�aH?
=q��\)��Q��!  C�^�                                    BxZo�  �          A	G�?�{��=q��  ��=qC�ff?�{��33��ff��C�`                                     BxZ~�  �          A�
>��У���G��  C��>��љ���  ���C�H                                    BxZ�J  �          A��>�  ��
=��  �L�HC�^�>�  ��Q���
=�Kp�C�Z�                                    BxZ��  "          A��>8Q������ff�d�C�  >8Q�������p��c  C�q                                    BxZ��  �          A�\    �������U�C���    ��=q�����T{C���                                    BxZ�<  T          AQ�>aG���
=����7�\C�q>aG��������
�5�HC��                                    BxZ��  "          @�\)���R��  �ٙ��gC�⏾��R�����أ��f
=C���                                    BxZֈ  �          @���ٙ��޸R��=qL�Ca�H�ٙ���ff����Cb�{                                    BxZ�.              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZ��              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZz              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZ   	S          @��þ������\¢8RCu33����{��=q¡G�Cv��                                   BxZ�  �          @�z�L�ͽL������°aHCB녾L�ͽ�����¯��CQ��                                   BxZ.l  �          @�?�p��������C�@ ?�p����������C�4{                                   BxZ=  "          @�{@ff�S�
��  �h{C��3@ff�W���
=�f=qC�q�                                   BxZK�  �          @���@�R=�����p�33@!G�@�R<���p�8R?:�H                                    BxZZ^  
�          @���?�33�C33�������C�P�?�33�G����
�}ffC��                                    BxZi  "          @���?�(��>{�љ��|�RC�'�?�(��B�\�У��z�C���                                    BxZw�  �          @�>�z��p�����H�l��C�4{>�z��u��ə��jz�C�*=                                    BxZ�P  
Z          @��>u�`�����
�w�HC��q>u�e�ҏ\�u\)C��3                                    BxZ��  �          @���?�33�?\)�ۅ�\C��\?�33�Dz��ڏ\�~C�Ff                                    BxZ��  �          @�ff?0���,������C�5�?0���1���(�8RC�H                                    BxZ�B  �          @�z�@ ���[���  �j33C�0�@ ���`  �θR�g�
C���                                    BxZ��  �          @�z�?�
=��(���Q��G�C��?�
=��{��{���C�H                                    BxZώ  �          @��
@������
=��
C�G�@���G������33C�,�                                    BxZ�4  T          @��
@�Q���=q��=q� �RC�*=@�Q����
��  ���HC�f                                    BxZ��  �          @�ff@�����z��l(���G�C���@�����{�hQ�����C��f                                    BxZ��  "          @�33@��\�����a���=qC���@��\��{�^{���
C�j=                                    BxZ
&  !          @��H@�=q��33�O\)��Q�C��3@�=q��z��J�H��C��{                                    BxZ�  T          @�\@���_\)�P  ��=qC��f@���a��L�����HC��)                                    BxZ'r  �          @��@��
����fff��33C��q@��
�(��dz���
=C�Ǯ                                    BxZ6  �          @���@:=q��ff�G��o�C�Z�@:=q��\)���c�C�P�                                    BxZD�  �          @�
=@4z���Q쿦ff�(�C��
@4z����ÿ����  C��                                    BxZSd  �          @���@ ����(����\��C��@ �����Ϳ�
=���C��                                    BxZb
  �          @�ff@
�H��(��Q��ÅC��f@
�H���K����C��
                                    BxZp�  �          A ��?�p���ff����� {C���?�p�����������C��                                    BxZV  �          @�
=?������?���ARffC��?����  ?��\A_33C��
                                    BxZ��  
�          @���@`������@XQ�A�p�C��=@`����33@]p�A�\)C��                                    BxZ��  T          @�\)@_\)���@8��A���C�� @_\)��Q�@>�RA��RC���                                    BxZ�H  
�          @�{@;���=q>�\)@�C�C�@;����>Ǯ@9��C�Ff                                    BxZ��  �          @��
@e���Ϳ�=q�B�HC��@e��p���p��6=qC���                                    BxZȔ  T          A ��@I����ff�Tz���(�C���@I����  �N�R�ˮC��R                                    BxZ�:  
�          A�H?����z��dz���{C�u�?����{�]p����HC�h�                                    BxZ��  �          A�R?+����S33��33C��
?+���\)�K���C���                                    BxZ�  
�          A�\=��
���R�&ff���C�G�=��
��  �{����C�G�                                    BxZ,  �          A �ÿ(����  ��p��dz�C��{�(�����ÿ����UG�C��
                                    BxZ�  "          @�\)?�{���H�Q��îC�o\?�{��z��J=q��(�C�e                                    BxZ x  T          @�R?��R?k���z�#�A�\)?��R?Q�����A�Q�                                    BxZ/  �          @���@G��Q�����qC��\@G��   ��\)L�C�)                                    BxZ=�  "          @�  ?�33�7���(��}C���?�33�?\)�ڏ\�zp�C�C�                                    BxZLj  �          @��?��
���R�����ffC�1�?��
��\)�xQ���HC�+�                                    BxZ[  �          @���?�������?��HA=�C��f?�����Q�?�=qAM��C���                                    BxZi�  �          @�\)?����
=����(z�C���?����=q���R�$�C�w
                                    BxZx\  �          A�H�W
=������33�g�HC��)�W
=�����أ��cC��                                    BxZ�  �          @��>��R��z�B�\��z�C�,�>��R���Ϳ�R���C�+�                                    BxZ��  "          A�H?xQ����?޸RAF=qC���?xQ���=q?��AV�RC���                                    BxZ�N  "          A(�?�������@�
A��\C��f?������@p�A��HC��                                    BxZ��  
�          A?�
=��@(��A���C��?�
=��(�@1�A��
C��                                    BxZ��  0          A�?ٙ����@�A}p�C�
?ٙ���=q@�A�
=C�                                      BxZ�@  F          A=q@ �����H@�AqG�C�1�@ ������@�A���C�:�                                    BxZ��  
�          Ap�?��
��{@qG�A�C��3?��
��@y��A�(�C��                                    BxZ�  �          A��@:�H���@���BC��)@:�H���R@���B�C��                                    BxZ�2  �          A�@������@�\A�z�C�3@���Ϯ@
=qA���C�"�                                    BxZ
�  �          A=q@���љ��QG����
C��@���Ӆ�H����ffC��f                                    BxZ~  T          A
=@��H��ff��(��(�C��@��H��G������C���                                    BxZ($  �          A{@�
=���H�fff���C��q@�
=����`�����
C�ff                                    BxZ6�  "          @��R@��H�e�@  ����C�� @��H�fff�.{��ffC��{                                    BxZEp  "          A
=A (��(�������=qC�XRA (��p����\���C�Ff                                    BxZT  �          Az�@����S�
?��A=qC��{@����Q�?�33A��C��                                    BxZb�  �          A ��@��H�{�?��A�\C��@��H�z=q?�A#�C�"�                                    BxZqb  �          @��R@ۅ�o\)?fff@љ�C��{@ۅ�n{?xQ�@��HC���                                    BxZ�  
�          @��R@�G����\?�\)Ap�C�aH@�G����?�p�AffC�p�                                    BxZ��  �          @�z�@��
���\?�AaG�C�9�@��
��G�?�An�HC�T{                                    BxZ�T  �          A�@�  ��Q쿬���  C��H@�  ���ÿ��H�  C���                                    BxZ��  T          A�R@]p���׽�Q�(�C��R@]p����=u>�(�C��
                                    BxZ��  �          A��@^{���333���HC�p�@^{��Q����r�\C�k�                                    BxZ�F  T          A(�@mp��޸R�����C��@mp���Q��33���C��{                                    BxZ��  T          A�@p���(���33��RC�.@p����R��  �z�C�                                      BxZ�  T          A�?��
�ȣ���  �{C��f?��
���
���
�z�C���                                    BxZ�8  T          A�R����\)���\�\C��f�����
����8RC��)                                    BxZ�  T          A�
@G����������;�
C��R@G���(������7��C�t{                                    BxZ�  T          AG�@Y�����H������HC��@Y������G����RC���                                    BxZ!*  
�          A��@�=q�\�
�H�lz�C�>�@�=q���
�33�^�\C�%                                    BxZ/�  F          A@��H��p����R�%�C��@��H��ff��\)���C���                                    BxZ>v  �          A\)@�����Q����O\)C���@���������\�B�\C��\                                    BxZM  �          A  @�  ��33��
�a�C�� @�  ��z��Q��T(�C�e                                    BxZ[�  �          AQ�@�������>�R���RC���@�����R�7���=qC�h�                                    BxZjh  �          A�R@�G���G�����\)C�/\@�G���=q��p���C�)                                    BxZy  "          A{@Å��녿xQ���C��R@Å���\�Y�����
C�˅                                    BxZ��  "          A
=@�ff��
=��\)�1G�C�L�@�ff��  ��  �$z�C�7
                                    BxZ�Z  
�          A(�@�(���ff������C�O\@�(���\)�h���ƸRC�C�                                    BxZ�   �          A{@�  ��p��u��G�C�T{@�  ��p�=�Q�?
=C�T{                                    BxZ��  �          A�@�������?��A(��C���@����׮?�
=A8Q�C���                                    BxZ�L  
(          A
=@������?���A
=C��)@�����(�?�G�A%C��                                    BxZ��  T          AQ�@�=q��p�?��@��C���@�=q���?@  @�Q�C�                                      BxZߘ  
�          A@��
�ָR��p��?�C�E@��
�׮�˅�/�
C�33                                    BxZ�>  T          A�R@��R��G��%���p�C��f@��R���H�(���G�C���                                    BxZ��  
�          A
=@u������`���ď\C��R@u���
=�W���(�C���                                    BxZ�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZ0              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZ(�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZ7|  X          A  @�����>�?n{C��H@�����>�\)?�p�C���                                   BxZF"  T          A��@�G���=q?\)@vffC��@�G��ᙚ?333@��
C��=                                   BxZT�  0          A��@��R��p�<��
>\)C�g�@��R��p�>#�
?��C�g�                                   BxZcn  
�          A=q@�G��(Q����
�&C��)@�G��.�R��=q�$ffC�&f                                    BxZr  �          A(�@��
�Fff�����+{C�j=@��
�Mp���
=�(��C��3                                    BxZ��  �          A
=@\)�x����p��+ffC��)@\)�\)���\�({C�y�                                    BxZ�`  T          A{@\����?@  @�C��)@\����?\(�@�\)C��f                                    BxZ�  T          A33@�{���
?
=q@p  C�� @�{���?#�
@�\)C���                                    BxZ��  �          A�@��H�J=q�X����\)C�/\@��H�N�R�U���33C��                                    BxZ�R  "          A=q@�z��:�H�E���{C��@�z��>�R�AG���z�C���                                    BxZ��  �          A	�@��33�H�����
C��@����Fff��
=C�K�                                    BxZ؞  "          A��@�  ��G���G��G�C�33@�  ��G�<��
>.{C�33                                    BxZ�D  
�          A
=@���أ�?�\)A�RC�~�@���׮?�G�A&{C���                                    BxZ��  T          Ap�@����(�?���A.=qC�XR@�����H?��HA=�C�g�                                    BxZ�  "          A33@Mp���@\)Av{C���@Mp���{@��A���C��=                                    BxZ6  "          A  @x����(�?�(�A�C��f@x�����H?�{A/�
C��3                                    BxZ!�  
�          A�@�
=��
=?=p�@�=qC�<)@�
=��ff?aG�@�  C�E                                    BxZ0�  
�          AQ�@�z��~{=�G�?=p�C��H@�z��~{>B�\?��
C��                                    BxZ?(  |          AffA�R��\)���H��RC���A�R��33��
=��
C���                                    BxZM�  J          Az�@˅��\)?L��@�C���@˅��
=?h��@�C��=                                    BxZ\t  �          A�@��ָR?Tz�@�z�C��3@���{?u@��HC��)                                    BxZk  
�          A��@���/\)?aG�@�C��@���.{?n{@�G�C��)                                    BxZy�  "          A	��@��vff>��?�  C���@��vff>k�?�G�C��R                                    BxZ�f  �          A33@��\��33�0����\)C�Ǯ@��\�Ӆ�\)�uC��H                                    BxZ�  �          AQ�@�����=q?s33@�Q�C�W
@�������?�ff@�{C�e                                    BxZ��  T          A�
@��R�Vff?��@��C�AH@��R�U�?�\)@�C�Q�                                    BxZ�X  �          A=q@��?0��@S33A�p�@��
@��?@  @Q�A��\@��                                    BxZ��  �          A@�ff�dz�?�Q�A33C��@�ff�c33?�G�A
=C��H                                    BxZѤ  "          A��@�=q����5��ffC�e@�=q��p��(���  C�\)                                    BxZ�J  �          A\)@X���h������S=qC�xR@X���qG��Ϯ�P
=C��q                                    BxZ��  T          A�@�G���������33C�^�@�G����H��p���C�&f                                    BxZ��  
(          A{@�����������  C�4{@���������{��C��)                                    BxZ	<  T          A  @����z��E�����C�!H@����ff�>{����C��)                                    BxZ	�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZ	)�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZ	8.              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZ	F�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZ	Uz              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZ	d               ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZ	r�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZ	�l              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZ	�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZ	��              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZ	�^              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZ	�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZ	ʪ              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZ	�P              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZ	��              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZ	��              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZ
B              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZ
�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZ
"�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZ
14              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZ
?�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZ
N�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZ
]&              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZ
k�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZ
zr              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZ
�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZ
��              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZ
�d              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZ
�
              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZ
ð              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZ
�V              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZ
��              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZ
�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZ
�H              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZ�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZ�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZ*:              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZ8�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZG�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZV,              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZd�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZsx              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZ�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZ��              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZ�j              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZ�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZ��              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZ�\              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZ�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZ�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZ�N              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZ�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZ�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZ#@              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZ1�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZ@�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZO2              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZ]�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZl~              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZ{$              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZ��              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZ�p              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZ�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZ��              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZ�b            A	�@�Q���  @�\A^=qC�t{@�Q����@�
A`��C�z�                                    BxZ�  �          A	@�33���\@(�A���C��@�33��=q@p�A�C�R                                    BxZ�  �          A
�R@�����@C33A��RC��R@���(�@C�
A�G�C��                                    BxZ�T  �          A�H@�����(�@Z�HA�\)C���@������@\(�A\C���                                    BxZ��  �          A=q@���ҏ\@&ffA��C���@����=q@'�A�33C�                                    BxZ�  "          A�@!���z�@z�Ag33C�(�@!���(�@ffAiC�*=                                    BxZF  T          A��@��\�W�@�
=B G�C��)@��\�W
=@�\)B �\C�f                                    BxZ*�  �          A�@��^�R@���Bz�C���@��^{@�G�B�RC��
                                    BxZ9�  T          AG�@����{@[�A�\)C��@����@\(�A�  C��=                                    BxZH8  �          Az�@�  ����@b�\Aʏ\C�S3@�  ����@c33A�33C�U�                                    BxZV�  
�          A��@|(��У�@Q�A�ffC��
@|(���Q�@R�\A��HC���                                    BxZe�  T          A��@����(�@%�A��C���@����(�@%�A��
C��=                                    BxZt*  �          Ap�@z�H��Q�?=p�@��C�.@z�H��  ?@  @�{C�.                                    BxZ��  �          A{@`����=��
?��C�y�@`����=��
?�C�y�                                    BxZ�v  �          A�@_\)��Q�@�
Aep�C��{@_\)��Q�@�
Aep�C��{                                    BxZ�  �          Aff@J�H��\)@*=qA��RC��
@J�H��\)@)��A��\C��
                                    BxZ��  �          A=q?������R@G�A_\)C��\?�����
=@ ��A^�RC��\                                    BxZ�h  T          A�\@`  ���
@U�A���C�}q@`  ���
@Tz�A�=qC�|)                                    BxZ�  T          AQ�@���
=@HQ�A��HC���@���
=@G�A�=qC��3                                    BxZڴ  T          A  @=p���Q�@K�A�z�C�
@=p����@J�HA��C�{                                    BxZ�Z  �          A�
@I����G�@��Aw
=C�Q�@I����@\)AuG�C�P�                                    BxZ�   	P          A\)@3�
��
=?���AUC��)@3�
��\)?�
=AS�C���                                    BxZ�  	          A\)?�����
=@Q�AiC���?�����\)@
=Ag33C���                                    BxZL  �          A{?��� z�@�
Ad(�C��?��� ��@�\Aa�C��                                   BxZ#�  �          A�
@Z�H��ff@Q�A�Q�C��3@Z�H���R@P  A���C���                                   BxZ2�  T          A�@��\����@���B
C���@��\���@�(�B
�C��H                                    BxZA>  �          A
=@����G�@��B33C�4{@�����
@��B  C�)                                    BxZO�  �          A�R@�(����@�{B!=qC�t{@�(���@�B ��C�XR                                    BxZ^�  �          Ap�@/\)�Å@�
=B
=C��@/\)��(�@�{B��C��                                    BxZm0  �          @�\)@��ڏ\@b�\AӅC�K�@���33@`  A�
=C�E                                    BxZ{�  �          Ap�@���\)@XQ�A��C�o\@���  @UA�z�C�j=                                    BxZ�|  �          Ap�>�Q����
@��A�Q�C�\)>�Q���(�@��A�\)C�Z�                                    BxZ�"  �          AG���
=���
�Ǯ�333C�����
=���
��G��L(�C���                                    BxZ��  "          @�=u��=q?�\)A�RC�<)=u��=q?���@�Q�C�<)                                    BxZ�n  �          @��\>L�����?��RApz�C���>L����?�
=Ai��C���                                    BxZ�  �          @�������@�A��RC�������p�@{A�
=C��3                                    BxZӺ  T          @�  �b�\�׮?��\A33CrG��b�\��  ?��HAz�CrQ�                                    BxZ�`  �          A z������  ?�z�A"�RCk�\�����Q�?���A  Ck�)                                    BxZ�  	>          A��������H?#�
@�ffCc�)������H?
=@�=qCc��                                    BxZ��  
|          A�����R��Q�?�G�A'�
Cf5����R�ȣ�?���A ��CfE                                    BxZR  �          A��=q��(�?��
AD��Cg���=q��z�?��HA=��Cg�
                                    BxZ�  �          A�\�����?=p�@��\C^:������?.{@�C^E                                    BxZ+�  �          A���p����
@#33A��C��H��p���z�@{A��RC��f                                    BxZ:D  �          A녿��H��R@ ��A�33C������H��@�HA�(�C��                                    BxZH�  �          A zῈ����  @'�A��HC��\��������@"�\A��C��3                                    BxZW�  �          A�?�  ��G�@s33A���C��?�  ���H@mp�A�p�C�H                                    BxZf6  �          A@.�R����@��B{C��f@.�R���@��HB�C��f                                    BxZt�  �          @�(�@���#�
@j=qB�RC��
@����Q�@j=qB�C�j=                                    BxZ��  �          A�H@�33@5�@��HA�A�  @�33@1G�@�(�A�=qA���                                    BxZ�(  �          A ��@�  ���@�
=B�HC��@�  ����@�
=BC�}q                                    BxZ��  �          A
=@ٙ��L��@�
=B�\C���@ٙ��^�R@��RB(�C�^�                                    BxZ�t  �          A��@����
@��\B	�C���@����@��\B	��C�]q                                    BxZ�  �          A@�p�>B�\@�p�A�ff?���@�p�>�@�p�A��\?�ff                                    BxZ��  �          A�H@�\)=�G�@��RA���?^�R@�\)=#�
@��RA��H>���                                    BxZ�f  �          A
=@��
?^�R@c�
A��H@߮@��
?Q�@dz�A�@�G�                                    BxZ�  �          A\)@�=q?��@J=qA��A[�
@�=q?޸R@L(�A�AU��                                    BxZ��  �          A
=@�  ?�=q@Tz�A�\)Ab=q@�  ?�\@VffA�33A[�                                    BxZX  �          Aff@�\)?�(�@EA�G�As�
@�\)?�@HQ�A�\)Amp�                                    BxZ�  �          A�\@�{@�R@X��Ař�A��@�{@�H@[�A�Q�A��                                    BxZ$�  T          A(�@޸R?��@�Q�A�Ag
=@޸R?�(�@���A뙚A]                                    BxZ3J  �          A\)@�z�?��R@���A�
=A#�
@�z�?�z�@�G�A��\Ap�                                    BxZA�  �          Ap�@�Q�<�@�\)B�>�\)@�Q�u@�\)B�C��)                                    BxZP�  �          AG�@��H���@�p�B  C�'�@��H��\@��B�C��R                                    BxZ_<  �          A z�@���p�@�=qB
��C��H@���=q@�G�B	�RC�q                                    BxZm�  �          A Q�@����  @���BG�C�]q@����ff@�\)B�C��                                    BxZ|�  T          A (�@��Tz�@�Q�B	C���@��Z�H@�{BG�C�w
                                    BxZ�.  �          @��@����w�@��B��C��\@����}p�@�=qB��C���                                    BxZ��  �          @��\@��H��p�@��RB��C��@��H��Q�@��BffC��)                                    BxZ�z  �          @��@&ff���
@�  B  C��@&ff��\)@��
BQ�C���                                    BxZ�   �          @��@��H��@z=qA�33C�y�@��H����@s33A�p�C�7
                                    BxZ��  �          @�33@�
=���\@x��A�  C�.@�
=��p�@p��A�C���                                    BxZ�l  �          @��@����@�\)B�C��3@����R@��HB�\C��f                                    BxZ�  �          @��@,����@�ffB	33C�7
@,����G�@���B
=C�f                                    BxZ�  �          @�G�?�(���\)@�=qA��C�Y�?�(��ҏ\@y��A�=qC�E                                    BxZ ^  �          @��
@X����(�@~{A�\C���@X����\)@tz�A�ffC��                                    BxZ  �          @���@b�\��Q�@���A���C�˅@b�\���@y��A�\)C���                                    BxZ�  �          A (�@5��(�@w
=A�C��@5��\)@l(�A܏\C��R                                    BxZ,P  �          A�R?����{@��
A�p�C�w
?���ᙚ@|(�A�33C�ff                                    BxZ:�  �          A z�?�z����H@u�A��C��
?�z���{@h��A�Q�C��q                                    BxZI�  �          @�{@Q���\@��A���C�H�@Q���z�@�
Aup�C�5�                                    BxZXB              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZf�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZu�   "          A   @�33��G�@K�A�33C���@�33��(�@AG�A�\)C��)                                   BxZ�4  �          A=q@����=q@S�
A�(�C��\@����p�@H��A�C�p�                                   BxZ��  �          AQ�@�p���p�@[�A���C�0�@�p�����@QG�A�
=C���                                   BxZ��  �          A�
@�{��Q�@/\)A�\)C���@�{���H@%�A�{C���                                   BxZ�&  �          Az�@�����
=@0  A��C��3@������@%�A�C��R                                   BxZ��  �          A(�@�ff���\@z�HA��HC��f@�ff��ff@n�RA��HC���                                    BxZ�r  �          A��@�33��{@n�RA���C��@�33���@a�A���C��{                                    BxZ�  �          AG�@�G����@(Q�A��C�T{@�G��\@�A��C�"�                                    BxZ�  �          AQ�@�����Q�?�z�AV=qC�:�@�����=q?�
=A<(�C�q                                    BxZ�d  �          A�@�
=��z�?�A<��C��q@�
=��{?�Q�A"ffC��H                                    BxZ
  �          A�@e��
=?�=qAQ�C���@e����?�=qA5p�C��=                                    BxZ�  �          A�\@��R�У�?�Q�AffC��=@��R���?u@�Q�C���                                    BxZ%V  �          A�@��H�޸R������C��q@��H��ff��\�b�\C��                                    BxZ3�  �          A  @��R��33��ff�陚C��f@��R��녿�ff���C��)                                    BxZB�  �          A=q@�p��ҏ\��Q��%�C��@�p��У׿�Q��AG�C�                                    BxZQH  �          A ��@|(��޸R���tz�C��q@|(���{�J=q��{C���                                    BxZ_�  2          A  @���������C���@���\�.�R��(�C���                                    BxZn�  v          A
=@�z����H?B�\@��\C��f@�z��Å?�@j�HC��
                                    BxZ}:  �          A�@�z����
>Ǯ@.{C�s3@�z���(�>\)?xQ�C�l�                                    BxZ��  �          A�
@��R�ʏ\?=p�@�p�C��q@��R��33>��H@Z=qC��\                                    BxZ��  �          A
=@�\)��ff?��@�
=C�\@�\)�Ǯ?O\)@�ffC���                                    BxZ�,  �          A@x�����?#�
@�=qC�n@x����\>�Q�@!G�C�e                                    BxZ��  �          A��@����ڏ\?!G�@�Q�C�@�����33>�33@\)C���                                    BxZ�x  �          Aff@5���\?�@���C�9�@5���33>�  ?��
C�4{                                    BxZ�  �          Aff@%�������uC�P�@%������Q��#33C�S3                                    BxZ��  �          AG�@>{��R>�Q�@%C���@>{��
==#�
>���C��R                                    BxZ�j  �          A��@333��ff@Q�Au�C��H@333��G�?�AP��C���                                    BxZ  �          A ��@dz���G�?�  A+�
C�q�@dz���33?�
=A�C�Y�                                    BxZ�  �          @�z�@p���ۅ�L�;�p�C�XR@p����33��Q��(Q�C�\)                                    BxZ\  T          @�(�@W
=���?�  @陚C���@W
=���?.{@�ffC��q                                    BxZ-  �          @��H@U���?Tz�@�=qC��R@U���?   @l(�C���                                    BxZ;�  �          @��
@�ff��{>u?�=qC�+�@�ff��ff�����C�(�                                    BxZJN  �          @��@�z�����?��AffC�%@�z����\?�ff@��RC�                                      BxZX�  �          @���@�����
@�{B=qC�N@����33@�\)B\)C���                                    BxZg�  �          @���@����ff@��B�\C�(�@�����@���B��C�~�                                    BxZv@  �          A z�@�z����H@��
B
=C�R@�z����@�(�B�C�u�                                    BxZ��  �          @��@��
��z�@�Q�B  C���@��
���H@���A�G�C�f                                    BxZ��  �          @���@~�R��
=@���Bz�C�3@~�R��{@���B��C�n                                    BxZ�2  �          @�
=@~{�J=q@��\B>C��q@~{�\(�@��B7\)C��                                    BxZ��  T          @��H@@  ��
=�Tz�����C�q�@@  �����z����C���                                    BxZ�~  T          @�@q���Q�>\)?��C��@q���Q�L�Ϳ�  C�3                                    BxZ�$  �          @��@XQ��ᙚ�8Q�����C��=@XQ���Q쿋����C��q                                    BxZ��  T          @�=q@qG����ÿ
=��=qC���@qG��׮�p����
=C��                                     BxZ�p  �          @�@%��ָR���
�@(�C���@%����
����m�C���                                    BxZ�  �          @�@H����(���p��x��C�)@H����Q��z����RC�S3                                    BxZ�  �          @�@�����{�G���ffC��=@�����zῌ���
=C���                                    BxZb  �          @�
=@8���׮�h����33C���@8�������� Q�C��3                                    BxZ&  �          @�?���33�޸R�\��C�g�?��׮����33C��                                    BxZ4�  �          @�Q�@vff��  ������C��3@vff����
=�1G�C��R                                    BxZCT  �          @��@���  ���\���RC�,�@���{��G��C�b�                                    BxZQ�  �          @�p�@�G����
�����(�C�0�@�G���G������?\)C�C�                                    BxZ`�  �          @�
=@   �љ���(��fffC�|)@   ��Q�O\)��G�C���                                    BxZoF  �          @�p�@��Ϳ8Q��(Q�����C��@��Ϳ\)�*�H��33C��q                                    BxZ}�  �          @��R@�G�����\)��Q�C��q@�G���  �ff���HC�>�                                    BxZ��  �          @�{=L����z�>��
@   C�.=L����z���s33C�.                                    BxZ�8  �          @�ff�#�
��z�?�z�A33C���#�
��ff?8Q�@�
=C��3                                    BxZ��  T          @�\)?=p���R?���A�C���?=p���Q�?#�
@��
C���                                    BxZ��  �          @�\)�\)���
?�{Ab�\C�s3�\)��
=?�33A)��C�u�                                    BxZ�*  �          @�ff?W
=��@i��A��
C��{?W
=��z�@N�RA�\)C�xR                                    BxZ��  
�          @���?��
��?�33AF{C��)?��
��ff?�
=AG�C��=                                    BxZ�v  �          @�G�?�G���?��A{C��?�G����?O\)@���C���                                    BxZ�  �          @��\?�G�����?�
=A��C���?�G���33?333@��C��{                                    BxZ�  �          @�{?5���@`��A��C�33?5����@EA�p�C�R                                    BxZh  �          @��
?u��Q�@��BC��\?u��G�@��HB�
C�\)                                    BxZ  �          @�\)?�(���Q�@uA��C��?�(���  @[�A���C��q                                    BxZ-�  �          @���@��ʏ\@u�A���C�@ @���=q@Y��Aљ�C��                                    BxZ<Z  �          @�Q�?����  ?�\)A#�C�U�?����\?\(�@�z�C�J=                                    BxZK   �          @�p�>�
=��?�Q�AQ�C���>�
=���?.{@��C���                                    BxZY�  �          @��@o\)��33@�(�B�
C���@o\)��z�@��Bp�C���                                    BxZhL  �          @�Q�@\(���\)@;�A��
C��R@\(���p�@   A�{C���                                    BxZv�  T          @���@-p��أ�@G�Aw�C��=@-p���z�?��A;�
C��)                                    BxZ��  �          @�{@��
��  @��HB�C���@��
���\@���BQ�C��q                                    BxZ�>  �          A Q�@����
=@	��A���C��H@�����?�\AS�C�`                                     BxZ��  �          A (�@�\)�n�R@n�RA�G�C�w
@�\)�\)@]p�A���C��3                                    BxZ��  �          @��@��
�=p�@��Bp�C��@��
�S�
@��\BG�C��                                    BxZ�0  �          A ��@�����{@��A�Q�C�z�@�����\)@l��A�
=C���                                    BxZ��  �          A ��@��\����@�z�A�G�C���@��\��33@s�
A�C��H                                    BxZ�|  �          @��@��\���@�Q�BQ�C��@��\���R@z�HA�C���                                    BxZ�"  T          @��R@�����@�  B{C���@���z�@��B(�C���                                    BxZ��  �          @��@�����H@��Bp�C�5�@�����R@�\)B�\C��
                                    BxZ	n  �          @��@�  �O\)@�p�B?�\C�\)@�  �k�@���B4p�C��\                                    BxZ  �          A ��@�
=��Q�@�=qB{C�]q@�
=��=q@��RB{C��R                                    BxZ&�  �          A ��@�ff�|(�@���B�C�3@�ff��Q�@}p�A�Q�C���                                    BxZ5`  �          Ap�@�
=���@u�A�(�C�&f@�
=����@_\)AΣ�C�8R                                    BxZD  �          A{@��R����@:�HA���C�  @��R���
@"�\A�=qC�]q                                    BxZR�  �          A (�@�������@R�\AǅC��f@�������@;�A�Q�C�H                                    BxZaR  �          A Q�@�����
@S�
AŅC�L�@�����
@5A�ffC��3                                    BxZo�  �          A ��@�����ff@*=qA�Q�C��\@�������@	��Aw�C�C�                                    BxZ~�  �          Ap�@���ff@UA��
C�u�@���ff@:=qA���C��f                                    BxZ�D  �          A z�@�����G�@l��A�\)C���@������\@QG�A�=qC��                                     BxZ��  �          AG�@������R@FffA�\)C���@�����ff@+�A�33C��                                    BxZ��  T          A z�@���z�=�?c�
C���@����
��(��G
=C��H                                    BxZ�6  �          @�
=@�{��  ?�(�Az�C�(�@�{�ҏ\?(��@���C���                                    BxZ��  �          @�{@������@G
=A��C�^�@������@+�A�C��=                                    BxZւ  �          A=q@�ff���\@\(�AʸRC��{@�ff��33@?\)A���C�3                                    BxZ�(  �          A33@�  �y��@aG�A�z�C�@�  ��{@J�HA��HC��                                    BxZ��  �          A�H@���N�R@��B ��C��f@���e@|(�A홚C�Y�                                    BxZt  �          AG�@�����p�@Dz�A�ffC�o\@�����p�@&ffA��C�                                    BxZ  �          A=q@������?�AR�HC��3@����љ�?��\Az�C�h�                                    BxZ�  �          Ap�@�(�����@   A��\C��@�(���
=@�\Am�C��\                                    BxZ.f  �          @��R@�������?���A0  C�\@�����z�?fff@ڏ\C���                                    BxZ=  �          A ��@�Q���\)?\(�@ȣ�C���@�Q�����>�z�@��C��{                                    BxZK�  �          AG�@tz���{��
=��
C�ff@tz���녿��P��C��                                    BxZZX  �          Ap�@�G���ff?.{@�  C�q�@�G����>��?���C�XR                                    BxZh�  T          Ap�@�  ����>��H@`��C�^�@�  ��G�        C�O\                                    BxZw�  �          A z�@�G���{�:�H��\)C��3@�G����H��ff��C��=                                    BxZ�J  �          A Q�@�����
=�����4��C���@��������	���x��C��                                    BxZ��  �          @�\)@q���\)�,(�����C�q@q���ff�R�\���HC��                                    BxZ��  �          @�33@�Q����  ��\)C�K�@�Q���{�5���33C���                                    BxZ�<  �          @�@�������!����\C�Z�@����Q��E����C��q                                    BxZ��  �          @��@��
���H�W
=��
=C��f@��
����x������C���                                    BxZψ  �          @��@�Q�����&ff��(�C�G�@�Q����\�C�
��G�C��                                    BxZ�.  �          @�@����33��ff��C��R@����z�����ffC�aH                                    BxZ��  �          Aff@$z��u�����]G�C���@$z��I����Q��o�C���                                    BxZ�z  �          A Q�@ �׿Y�����.C�~�@ �׽�Q����R�qC��q                                    BxZ
   �          @�z�?��R�8Q���(�¡aHC��?��R?
=����AʸR                                    BxZ�  �          @�  ?��H���H?�
=A,  C��3?��H��{?.{@��
C�z�                                    BxZ'l  �          @�\)@���>�z�@C��@���33���H�aG�C�Ǯ                                    BxZ6  �          A ��@=q��\?�=q@��RC���@=q����>���@�C���                                    BxZD�  �          @�\)@Z=q��z��=q���HC��3@Z=q�˅�E����RC�
                                    BxZS^  �          @��@h�����R�hQ��ۮC��R@h��������\)��C���                                    BxZb  �          @��@�����=q��\)��C�%@�����Q�xQ���(�C�Ff                                    BxZp�  �          @��@��\��z�����Q�C�Ǯ@��\�љ����R�{C���                                    BxZP  �          A ��@l(���\)�У��:=qC���@l(��أ��ff���
C�J=                                    BxZ��  �          A (�@^�R���H��R��p�C�}q@^�R��=q�<(����RC��R                                    BxZ��  �          A��@j�H���H��(��33C���@j�H��p���(��b=qC���                                    BxZ�B  �          @�
=@%���z���H���HC�g�@%���z��&ff��Q�C��{                                    BxZ��  �          @���@N{��  @�z�B�C�,�@N{��
=@�Q�A���C�,�                                    BxZȎ  �          @��@@  ���@\)A��C��q@@  ���?�p�A0  C��\                                    BxZ�4  �          @��@|����=q?��A5C��H@|����ff?O\)@�ffC�B�                                    BxZ��  �          @�
=@fff���?�z�Ak
=C�� @fff��\)?���A�RC���                                    BxZ�  �          @�ff@9�����H@?\)A�z�C���@9����(�@\)A���C�o\                                    BxZ&  �          A ��?��H���R@�  B�C��?��H��@s33A�{C�|)                                    BxZ�  �          A�
@�=q���H@Q�A�33C���@�=q��=q?�33A9�C�j=                                    BxZ r  �          A\)@R�\��p�@3�
A�33C���@R�\��ff@ ��Ad(�C�H�                                    BxZ/  �          A�\@l�����@�Ax��C�3@l����?���AQ�C���                                    BxZ=�  �          A{@Dz���ff?�\)AW33C��@Dz����
?��
@�(�C�Q�                                    BxZLd  �          A@tz���\)?��A��C�Y�@tz����H?   @a�C�+�                                    BxZ[
  �          @�ff@vff����@4z�A��C��@vff���@z�As�C�5�                                    BxZi�  �          @�z�@Fff��G�?�  A0Q�C��H@Fff��p�?+�@��
C���                                    BxZxV  T          @�?��H��(�?���@��C�n?��H���R>#�
?�
=C�aH                                    BxZ��  �          @�p�?�����  >k�?��C�P�?�����
=�0������C�U�                                    BxZ��  �          @�{@\)��������RC���@\)��׿���RC��                                    BxZ�H  �          A ��@`  ��\)?�\)A;�
C�P�@`  ��(�?G�@��HC�{                                    BxZ��  �          @���@"�\���?(��@���C�xR@"�\���u��  C�p�                                    BxZ��  �          A ��@,(���=q=#�
>�=qC��=@,(���Q�aG�����C�ٚ                                    BxZ�:  �          @�ff@E���þ��R��C���@E��{�����\)C��                                    BxZ��  �          A�@�ff��(���Q����C�Ф@�ff��
=��
=�z�C��
                                    BxZ�  �          A (�@����
=@�\AqC��3@����p�?��AC�|)                                    BxZ�,  �          A�@%���ÿ\)��(�C���@%������H�-�C��{                                    BxZ
�  T          A (�@�\������\�5��C�Q�@�\��  ����O�\C�#�                                    BxZx  �          A ��?У��.{��R��C�s3?У׿����  =qC�q                                    BxZ(  �          A (�?��8Q���Q��y��C�{?���\�ڏ\p�C�Ф                                    BxZ6�  �          @�{?�G������\)�]�C�+�?�G��\����
=�xC��                                    BxZEj  �          A (�?��������
�V{C�c�?��c�
��(��q{C��{                                    BxZT  �          A��@*�H��33��Q��+�C�� @*�H��=q���E�RC���                                    BxZb�  T          A�þ\��Q���\�e�C��)�\�c�
��33�{C���                                    BxZq\  �          Az�+�����R�h�\C��{�+��]p���
=�C��f                                    BxZ�  T          A\)�:=q�l(���z��^�HCg�3�:=q�0����=q�u
=C_s3                                    BxZ��  �          A��@�R�33����#\)C�l�@�R��p��&ff��33C���                                    BxZ�N  �          A	>aG��{��
=�   C��)>aG��ff�G��x��C��H                                    BxZ��  �          A	?�Q���þ��Q�C��?�Q���H��=q�,��C�!H                                    BxZ��  �          A�ÿ�G��{>�\)?�Q�C��ῡG��p��O\)���C���                                    BxZ�@  �          A���Q���\?=p�@���C�Q��Q��
=��Q��p�C�W
                                    BxZ��  �          A{��p���<��
>#�
C��Ϳ�p��   �����{C��H                                    BxZ�  �          A�<#�
��=q@(��A��C�
=<#�
���
?�Q�AU�C�
=                                    BxZ�2  �          A��@
=����@Y��A�G�C�3@
=��@Q�A��C��                                    BxZ�  �          Az�@����Q�@q�A�33C��\@����R@1�A�33C�l�                                    BxZ~  �          A��?�ff��{@�G�A�{C��?�ff��
=@U�A���C���                                    BxZ!$  �          Aff?����(�@�Bz�C�~�?����  @~{A�z�C�
=                                    BxZ/�  �          A��@����ff>�ff@C33C�/\@����{����l(�C�33                                    BxZ>p  �          A�H@�
=��G��+���G�C�f@�
=���
���H�1�C�P�                                    BxZM  �          Aff@����׿h����{C�h�@����=q���H�Mp�C��H                                    BxZ[�  �          Aff@��\���ÿ���C�C��\@��\��ff�6ff���C���                                    BxZjb  �          A��@�����Q�h����(�C�>�@�����녿�{�@z�C��=                                    BxZy  �          A  @���ڏ\�@  ��\)C��)@���ʏ\�|(���  C���                                    BxZ��  �          A(�@����z��Q�����C���@����33��{���C���                                    BxZ�T  �          A��@�(���=q�z=q��(�C��f@�(���{��  ��z�C�T{                                    BxZ��  �          A��@��������p��
�C�#�@���������� �C���                                    BxZ��  T          Ap�@��R���
��G��<�C���@��R�G�����Q�\C�|)                                    BxZ�F  �          AG�@hQ��R�\���d�C��@hQ����� (��x��C���                                    BxZ��  �          A�@�(������a�����C��\@�(���
=�������HC�}q                                    BxZߒ  �          A=q@������
����_�
C�T{@�����\)�>{��(�C�W
                                    BxZ�8  �          Aff@ָR��ff�Z=q��G�C��@ָR��(������ׅC��\                                    BxZ��  �          A@У���(����H��33C��R@У���
=������ffC��                                    BxZ�  �          A=q@�����G���
=�=qC���@����j�H���H�(p�C�|)                                    BxZ*  �          A=q@�  ��=q����$�
C�@�  �G���G��7�RC�7
                                    BxZ(�  �          A��@��O\)���
�=33C��@���������MQ�C��H                                    BxZ7v  �          A
=@��
>�������xG�@G�@��
?�������r33A�ff                                    BxZF  �          A�@r�\?xQ��p�Q�Aep�@r�\@(����H�s=qA�                                      BxZT�  �          A�\@j=q?
=q� Q�z�A{@j=q?�G����H�zz�Aͅ                                    BxZch  �          A�@��R=�G���p��N?��@��R?���=q�J�AA��                                    BxZr  �          A�\@u�E�����zp�C�]q@u>�����H�|
=@��
                                    BxZ��  �          A�
@R�\��G���{�2p�C��
@R�\��������O\)C���                                    BxZ�Z  �          Az�@tz�J=q���H�{
=C�#�@tz�>������
�|�H@�
=                                    BxZ�   �          A	�@�G��N{�#33����C�
=@�G��1��AG�����C���                                    BxZ��  �          A�@��\���@<��A��C�%@��\��  ?��RAZ=qC�9�                                    BxZ�L  �          A	G�@�=q��p�?�
=AO�C��f@�=q���?�=q@�Q�C�E                                    BxZ��  �          A	p�@�33�{�@U�A��RC��@�33���@*=qA��\C���                                    BxZؘ  �          A	�@�z���\)@\��A���C��)@�z��ƸR@\)A�
=C�xR                                    BxZ�>  �          Az�@����@�33B��C�9�@�����@u�A��HC�1�                                    BxZ��  �          A	G�@����
=?��RAV�HC��@����ff?k�@�C�                                    BxZ�  �          A	@��
���@�A�\)C�3@��
��
=?���Ap�C�xR                                    BxZ0  �          A	G�@����z�@C�
A�{C�˅@���ᙚ?�Q�ARffC��                                    BxZ!�  �          A�
@(��� ��� ���T��C�{@(����(��S33���C��\                                    BxZ0|  �          A�@(�������-p�C�E@(����H�>{����C��                                    BxZ?"  �          A
{@|����Q�8Q���33C��q@|����G������S\)C�8R                                    BxZM�  �          A
{@c�
����{�C�s3@c�
��33�'
=���C��                                    BxZ\n  �          A
=q?�p���\)��\�|(�C�E?�p����e����C���                                    BxZk  �          A
�H��Q���\�4z���ffC�� ��Q���\��z����C���                                    BxZy�  �          A(��:�H���Q��|��C�~��:�H��33�o\)���C�U�                                    BxZ�`  �          A�
�z�H���ٙ��4��C��
�z�H���G
=��G�C���                                    BxZ�  �          A	p�@*�H�G���Q��RC�'�@*�H��ff���H���C�K�                                    BxZ��  �          A	p�@�R�ff?!G�@�  C�w
@�R�=q�B�\���HC�y�                                    BxZ�R  �          A
�H@   �\)>���@)��C�|)@   �ff��G���\)C���                                    BxZ��  �          A
=q@�33?Q�@�
=C��R@�������  C��3                                    BxZў  �          A��@>�R��\@HQ�A���C�3@>�R��Q�?�=qAFffC��                                    BxZ�D  �          A(�@8Q��ʏ\@�B\)C�=q@8Q���\@p��A�C��                                    BxZ��  �          A=q@"�\���@��B1=qC��\@"�\��
=@��BC�                                      BxZ��  �          A  @�����=q?O\)@�G�C�k�@������
�������C�Q�                                    BxZ6  �          A	G�@��H�ȣ������G�C�8R@��H��  �\(���p�C�o\                                    BxZ�  �          A	G�@���\)�1����C�!H@������q���Q�C��
                                    BxZ)�  �          A�H@��
����e�˅C�S3@��
���H������C�N                                    BxZ8(  �          A�@��H���
�W���z�C�n@��H��������=qC��                                    BxZF�  �          A\)@��\�����\)�ᙚC�AH@��\�������C�l�                                    BxZUt  �          A  @����������HC�P�@�������{��RC�q                                    BxZd  �          Aff@�\)��p��j=q�ӮC���@�\)��{������RC�ٚ                                    BxZr�  �          A�@�z�����?B�\@���C�Ff@�z������\)��C�
                                    BxZ�f  �          A�H@������>u?���C�"�@������333��  C�>�                                    BxZ�  �          A�H@߮���L�Ϳ�{C�{@߮��=q�����(�C�g�                                    BxZ��  �          A�R@�������?\(�@��C��@������
=�?J=qC�@                                     BxZ�X  �          A=q@�G���p�>�
=@7
=C��@�G���p����H�S�
C��                                    BxZ��  �          Ap�@�p����\>��H@XQ�C��\@�p���=q�
=q�mp�C��3                                    BxZʤ  �          A��@��R��G�=�G�?G�C��f@��R���R�s33���HC�
                                    BxZ�J  �          A33@��~{=�Q�?��C��{@��z�H�&ff��C��)                                    BxZ��  �          A�@���A녾u��{C���@���;��L����(�C��3                                    BxZ��  �          A
=@����
=�\)>��C�Q�@���녿@  ���HC��H                                    BxZ<  �          A
=@߮��>k�?�=qC�
@߮��z�&ff����C�33                                    BxZ�  �          A
=@ٙ���ff>Ǯ@*=qC���@ٙ���{�
=q�l��C�                                      BxZ"�  �          A�\@߮��p�>u?�z�C�!H@߮��(��#�
��33C�<)                                    BxZ1.  �          A(�@�G���>�G�@<��C�!H@�G���p������Q�C�*=                                    BxZ?�  �          A=q@�\)��?8Q�@�C��H@�\)��\)�k���ffC��                                     BxZNz  �          Aff@љ����?�=q@���C��@љ���
==�Q�?��C���                                    BxZ]   �          A�R@�{�5>aG�?��
C�� @�{�5�������C���                                    BxZk�  �          A�H@�ff�H�ÿ8Q���
=C��@�ff�<�Ϳ�����C��f                                    BxZzl  �          Az�@��\�W�>��?��\C�XR@��\�U�   �W
=C�s3                                    BxZ�  �          A�
@��R�]p�=�G�?G�C��@��R�Z�H�\)�qG�C��                                    BxZ��  �          A�\@��R�S33�B�\���
C�j=@��R�L�ͿQ���(�C���                                    BxZ�^  �          Aff@�\��p�?333@��C���@�\��\)���^�RC��3                                    BxZ�  �          A@޸R��?�{ANffC��H@޸R���R?�G�@޸RC��=                                    BxZê  �          AG�@���\��?W
=@�33C��@���c33>��?��
C�7
                                    BxZ�P  �          A�\A�ff���R�\C��)A��(��c�
����C�/\                                    BxZ��  �          A@��p  ?�@hQ�C��
@��r�\�aG����
C�~�                                    BxZ�  �          A=q@\����?�Q�AXQ�C��@\����?\(�@�C���                                    BxZ�B  �          A�R@��H��p�@\)A��\C��@��H��G�?�G�A33C�+�                                    BxZ �  |          A��@���^{����n�RC���@���R�\��p��p�C�xR                                    BxZ �  �          A��A�2�\>8Q�?�z�C��fA�1G��\�\)C��
                                    BxZ *4  �          A\)@�����@\)Ao\)C�XR@����
=?�\)A��C�Ff                                    BxZ 8�  �          A�@�=q���H?�p�A Q�C�3@�=q��  >���@	��C��                                     BxZ G�  �          A��@������?s33@�{C�3@������R=���?.{C��                                    BxZ V&  �          A�@�G��~�R?���@�C�w
@�G����
>�=q?�\C�\                                    BxZ d�  �          A@�=q�^{�#�
����C��@�=q�Y���8Q���=qC�C�                                    BxZ sr  �          A��A��p��A���ffC�}qA���ff�Z=q��p�C��\                                    BxZ �  �          A��A33���H�L����{C�C�A33���\�a���ffC���                                    BxZ ��  �          A��A	G��������o33C���A	G�����&ff��  C�}q                                    BxZ �d  �          A�Ap���p��mp�����C���Ap���G��xQ��̸RC�p�                                    BxZ �
  �          A�\@�ff�������z�C��3@�ff����ff��C�R                                    BxZ ��  �          A@�33�
=��\)�

=C�y�@�33���
����\)C��3                                    BxZ �V  �          A�@��R�׮?�\)AffC��H@��R��(��u����C�|)                                    BxZ ��  �          A�H@�Q���G�?G�@��C�Ff@�Q���녿����=qC�=q                                    BxZ �  �          Az�@�=q��=q�j=q���HC��@�=q��p������=qC�o\                                    BxZ �H  �          A(�@dz���33���1G�C���@dz��S33�У��U33C��)                                    BxZ!�  �          Aff@e�������\)�@G�C��@e��8���߮�b�C���                                    BxZ!�  �          @�
=@�=q��
=�6ff����C��3@�=q����|(���C��)                                    BxZ!#:  �          @�@�\)��@ ��At��C��f@�\)���?^�R@љ�C���                                    BxZ!1�  �          @��@�  �N{�)�����HC�.@�  �%�Q���C��                                    BxZ!@�  �          @��H?�?O\)��G��\A��R?�@����ff�BS�                                    BxZ!O,  �          @�=q@:�H�33��ff�{ffC�s3@:�H��\��\)p�C���                                    BxZ!]�  �          @���@'��}p���
=�{C��=@'�?z���Q�k�AH��                                    BxZ!lx  �          @�z�?�Q��p����
B�C��\?�Q�Q���\)ǮC��=                                    BxZ!{  �          @�
=?}p�?�=q���R��Bf��?}p�@L����{��B��                                     BxZ!��  �          Ap�?��?.{��p�¡��A��?��@Q����H��B��H                                    BxZ!�j  �          A=q��?�z���ffL�B��\��@Vff����=qB�B�                                    BxZ!�  �          A녾�p�@���\)�3B�z᾽p�@z=q��G��sp�B��
                                    BxZ!��  �          @���@vff��=q�W
=���C��R@vff�O\)�g��*�\C�{                                    BxZ!�\  �          @��@�Q���p�?�z�Aw33C�l�@�Q����?s33@�33C�g�                                    BxZ!�  �          @�@�=q�hQ�@H��A�{C�w
@�=q��\)@��A�\)C�ff                                    BxZ!�  �          @�{@љ�����@�A��RC��@љ���\)@�
A��C�
=                                    BxZ!�N  �          @��@ҏ\��?�  AE��C�,�@ҏ\�Q�?�ffA
=qC��                                    BxZ!��  �          @��H@�\)��z�>\@G
=C�H�@�\)��(�=�?s33C�                                    BxZ"�  �          @ᙚ@���Q쾞�R�\)C��@�����\)���HC���                                    BxZ"@  �          @�\@ۅ�W
=�����5�C���@ۅ��\����IG�C��                                    BxZ"*�  �          @�(�@ʏ\?}p������
A=q@ʏ\?��R�33���AS�                                    BxZ"9�  �          @�
=@hQ�?k��
=��G�Ad  @hQ�?�\)�������A��                                    BxZ"H2  �          @\@p��@,(��@  �{B33@p��@R�\�z���G�B$�H                                    BxZ"V�  �          @���@C33@����1G���z�Be�\@C33@��׿�\)�nffBo                                    BxZ"e~  �          @��
@�z�@-p��$z����HAŮ@�z�@N{����{\)A�p�                                    BxZ"t$  �          @�z�@�=q>�(��Dz���=q@�=q@�=q?�\)�8�����HA'
=                                    BxZ"��  �          @�@��
@��~{���A�=q@��
@QG��Tz���  B��                                    BxZ"�p  �          @�ff@\?����*=q��
=Al  @\@
�H�{��A���                                    BxZ"�  �          @�=q@7�>�
=�
�H�33A�\@7�?h��� ���33A��R                                    BxZ"��  �          @��@J�H����?�G�Ac�C�#�@J�H��\)>��R@:�HC��)                                    BxZ"�b  �          @Ǯ�#�
��
=@���B(
=C����#�
����@8��A噚C���                                    BxZ"�  �          @�z�?�G����H@��\B?�C��3?�G�����@|(�Bz�C���                                    BxZ"ڮ  �          @�z�@"�\��
=@u�B��C���@"�\���R@ ��A���C��                                    BxZ"�T  �          @�z�@�z���  @'
=A�G�C��H@�z���
=?���A1��C�]q                                    BxZ"��  �          @�R@�������?���Ak\)C�+�@������?Tz�@�z�C�%                                    BxZ#�  �          @ۅ@����Dz�?h��@���C�AH@����L��>.{?�
=C��f                                    BxZ#F  �          @�{@�G���
?+�@��C�B�@�G��
=q>.{?�
=C��)                                    BxZ##�  �          @�
=@θR�У�?�p�AH��C���@θR��
=?��A�RC��=                                    BxZ#2�  �          @�\)@����\)@ffA���C��f@����?�33A��
C�~�                                    BxZ#A8  �          @�=q@�(��.�R?�@�Q�C�Ф@�(��1녽�Q�@  C���                                    BxZ#O�  �          @�  @���9�������  C��@���+����\�"{C��                                    BxZ#^�  �          @�(�@�\)@<������  A�33@�\)@\(���z��mG�BG�                                    BxZ#m*  �          @ٙ�@�G�@��\�Q���BAff@�G�@�(�>�33@EBB                                    BxZ#{�  �          @�33@vff@�33����
=BY{@vff@�{>�=q@�
BZ��                                    BxZ#�v  �          @Ӆ@��R@R�\�Q�����B��@��R@~�R�������B&�
                                    BxZ#�  �          @�Q�@Q�@�ff������BP  @Q�@�=q�u��BY(�                                    BxZ#��  �          @���@��@�{?�Q�A<(�B��@��@�  @��A�ffB}(�                                    BxZ#�h  �          @�R@��@2�\��ff�(�A�@��@n{�Y����G�B�
                                    BxZ#�  �          @�R@���@�G��=q��\)B�@���@�  ��{�-��B�                                    BxZ#Ӵ  �          @�R@���@l��������
B	�@���@���p��C\)B�H                                    BxZ#�Z  �          @���@��H��(����R�L\)C�Y�@��H?xQ���z��I
=A?�                                    BxZ#�   �          @�z�@@  �Dz���z��W�C�&f@@  ��ff�θR�y�RC�]q                                    BxZ#��  �          @���@��\)����~\)C��@������  ��C���                                    BxZ$L  �          @�G�?˅�
=q��RǮC���?˅?��R���
ffB                                    BxZ$�  �          @��@(Q�?����ٙ�\)B=q@(Q�@]p���(��ZBR�
                                    BxZ$+�  �          @���@p�@"�\��G��~z�BC�@p�@�(����R�O=qBwp�                                    BxZ$:>  �          @�Q�@\)@3�
�ۅ�yz�BMff@\)@�����R�I33B|(�                                    BxZ$H�  �          @���@G�@7
=���H�{��B[Q�@G�@��R��p��I�B�aH                                    BxZ$W�  �          @�@,��@�z���z����By�@,��@�  �.�R��Q�B��\                                    BxZ$f0  �          @�Q�@�@^�R���H�i��Bl33@�@�  �����6{B�                                    BxZ$t�  �          @��R?�p�@��H��ff�V�B�L�?�p�@�  ��\)�!�B�aH                                    BxZ$�|  �          @��?�@�������$  B��?�@ҏ\�aG����
B��                                    BxZ$�"  �          @�Q�?�  @�\)�w���B�Ǯ?�  @����~�\B���                                    BxZ$��  �          @�=q?�@љ���Q���33B��\?�@��H�\)��p�B�#�                                    BxZ$�n  �          @��@C�
@�
=�Fff����B�L�@C�
@�G������#
=B���                                    BxZ$�  �          @�{@   @�ff��\�v�HB�z�@   @��aG���B��                                    BxZ$̺  �          @�=q?�p�@�{�����E�B��3?�p�@�>��@   B�=q                                    BxZ$�`  �          @��R�@  @�=q��
=�J�HB�
=�@  @�{?���A$��B�                                    BxZ$�  �          @�  ��@�\��R��33B֞���@�?�G�A�RB���                                    BxZ$��  �          @�G��33@�z��G��VffB�Ǯ�33@�Q�?��A*ffB�\)                                    BxZ%R  �          @��ÿ�33@��H�u���
B�8R��33@�\?�G�@�
=B�=q                                    BxZ%�  �          @�Q����@��L����{BϽq����@�{?��A�RB��f                                    BxZ%$�  �          @�����z�@�׿��
��B�=q��z�@�G�?n{@�z�B�33                                    BxZ%3D  �          @�Q��^{@��=���?J=qB�\�^{@��?�A^�HB��H                                    BxZ%A�  �          @�\)����@���?Tz�@�33CaH����@��@  A�ffCz�                                    BxZ%P�  �          @�=q��z�@�
=�L�;�p�B�����z�@���?��RA9B�p�                                    BxZ%_6  �          @���g�@�
=>�\)@
�HB�p��g�@�?�Q�Ar�RB�R                                    BxZ%m�  �          @�ff��R@��?��HA�B�.��R@�33@AG�A�p�B�#�                                    BxZ%|�  �          @����\)@�Q�?   @j�HB�Q�\)@�@\)A�33B�Ǯ                                    BxZ%�(  �          @�  �(��@�ff>�G�@R�\B�Ǯ�(��@�=q@�A�Q�B�G�                                    BxZ%��  �          @����33@�\?k�@��HB�G���33@�\@7
=A�  B�z�                                    BxZ%�t  �          @�z�\@�ff?z�@��\B���\@�G�@   A�
=B�ff                                    BxZ%�  �          @�׿�@�R>�Q�@1�B��f��@ۅ@�RA�33B�Q�                                    BxZ%��  �          @��	��@�=q?��
@�ffB��H�	��@љ�@6ffA��B�ff                                    BxZ%�f  �          @��H�ff@�G�?�z�AS
=B�\)�ff@Å@X��A�\)B��f                                    BxZ%�  �          @�R���@�=q?޸RAl��B�#׾��@�(�@Z=qA�(�B�{                                    BxZ%�  T          @�G�@�\)>#�
�!G����R?��@�\)?Q������(�@��                                    BxZ& X  �          @�G�@��H>B�\��  �]�?�ff@��H?&ff����O\)@�
=                                    BxZ&�  �          @�  @��@1녿�
=�{A��H@��@?\)�����@��A���                                    BxZ&�  �          @�R@��\@P  @xQ�A��HA���@��\@ff@�33B{A�(�                                    BxZ&,J  �          @��@�z�@�=#�
>���B/p�@�z�@�\)?�z�A.�\B+                                      BxZ&:�  �          @�
=@��@�(����^�RB3p�@��@��?}p�@�p�B1�                                    BxZ&I�  �          @�(�@ȣ�@�녿���(�B	��@ȣ�@��R=L��>ǮB��                                    BxZ&X<            @��H@�ff?�zῘQ���AV�H@�ff?��5���HAv�H                                    BxZ&f�  
�          @�p�@�G�@5��z����A�=q@�G�@3�
>�@`��A���                                    BxZ&u�  
�          @��
@�G�@��?��A�\A�z�@�G�@ ��?�Ac�Av=q                                    BxZ&�.  "          @��@߮?�Q�@#�
A�p�Aw�@߮?�
=@?\)A���A��                                    BxZ&��  �          @�\)@θR?���@u�A홚A|Q�@θR?8Q�@�B��@���                                    BxZ&�z  T          @�Q�@�p�@=q@��B	33A�G�@�p�?��@�33B=qA.�H                                    BxZ&�   �          @���@�ff?�\@�ffB��A�{@�ff>�Q�@�  B*\)@i��                                    BxZ&��  
�          @��@�\)?�{@���B;�HAt(�@�\)�B�\@�{BC(�C��f                                    BxZ&�l  �          @�z�@��?���@tz�A�A�@���#�
@~{A�  C��R                                    BxZ&�  �          @��@�p�?�{@Q�AƏ\AQ�@�p�?(��@fffA�
=@�                                    BxZ&�  �          @�(�@ٙ�?5@y��A�p�@��R@ٙ��\@|(�A��\C�g�                                    BxZ&�^  �          @��
@���?+�@j�HA�\)@�G�@��;�Q�@n{A�=qC���                                    BxZ'  
(          @�ff@�=q��G�@�  BQ�C�H@�=q�˅@}p�A�z�C��{                                    BxZ'�  "          @�p�@�
=�B�\@e�A�p�C��=@�
=���H@O\)A�Q�C��)                                    BxZ'%P  �          @���@ᙚ��@N�RA�
=C�� @ᙚ�k�@FffA�z�C�K�                                    BxZ'3�  
�          @�G�@޸R=���@^�RA���?L��@޸R�^�R@XQ�A�C�t{                                    BxZ'B�  �          @�  @�z�L��@w
=A�{C���@�z��@_\)A�
=C�C�                                    BxZ'QB  
<          @��@\�
=@���B{C�k�@\�N{@c33A�z�C��                                    BxZ'_�  
�          @��H@�����@���B
\)C�w
@���b�\@]p�A�33C�+�                                    BxZ'n�  
�          @�\@���=q@��B��C��f@��/\)@r�\A�
=C��
                                    BxZ'}4  T          @�z�@�녿�
=@��RBffC�� @����@}p�A�z�C�Q�                                    BxZ'��  "          @�\)@�ff��
=@�{B&Q�C���@�ff�   @�p�B��C���                                    BxZ'��  
�          @��H@�G���@���B8z�C���@�G���p�@��B(�HC�H�                                    BxZ'�&  
�          @�z�@��>#�
@��\B=q?�\)@����\)@�{B�
C�xR                                    BxZ'��  T          @�@��׿�ff@�(�BQ�C���@����#33@��HB�RC�
                                    BxZ'�r  T          @�R@�\)�@��@H��A���C���@�\)�p��@
�HA���C��f                                    BxZ'�  
�          @�ff@��\�{@1�A�33C��@��\�I��?�(�A��
C��                                    BxZ'�  
(          @�  @�(��>{@I��A�33C���@�(��o\)@(�A��
C���                                    BxZ'�d  �          @�@�(��[�@N�RA��C��)@�(����R@Q�A���C�7
                                    BxZ(
  
�          @�@�
=�_\)@r�\A�ffC���@�
=��p�@*=qA�p�C���                                    BxZ(�  T          @�@�Q���{@<��A��RC�aH@�Q���33?�33AJffC�Q�                                    BxZ(V  �          @�  @�Q��vff@4z�A�
=C��@�Q����?�{AG33C��                                    BxZ(,�  T          @�(�@�p�����?��Ao
=C�
@�p����
?�@��
C�
=                                    BxZ(;�  T          @�
=@��R���?�ffA!G�C���@��R��녾aG���z�C�5�                                    BxZ(JH  �          @�@�(��O\)@�HA��\C�y�@�(��r�\?�33A@  C�g�                                    BxZ(X�  
�          @�Q�@��
>8Q�@G
=A��?�@��
�5@A�A�\)C���                                    BxZ(g�  �          @��H@���>�G����
�L��@x��@���>�(�=���?k�@q�                                    BxZ(v:  �          @�  @�\)��=q?�=qAxz�C�  @�\)����>�ff@o\)C���                                    BxZ(��  �          @�R@w����\@�A���C�Z�@w���  ?
=@��RC�e                                    BxZ(��  
Z          @�
=@XQ���Q�@G�A��HC��=@XQ����H>aG�?�(�C�f                                    BxZ(�,  	�          @��@Q논��
@n{BBG�C�ٚ@Q녿�=q@c�
B7G�C��)                                    BxZ(��  
�          @�
=@�p�@u@hQ�A���B�@�p�@(��@��\B!�A�\)                                    BxZ(�x  T          @�@x�ÿ�\)@�z�BT��C���@x���K�@�B1�
C�e                                    BxZ(�  
�          @���@P�׿��@��Bm��C���@P���,��@�Q�BL�RC�5�                                    BxZ(��  
(          @��H@s�
��  @�ffBY33C�B�@s�
�E�@���B6�
C���                                    BxZ(�j  �          @�z�@���!�@��B\)C�AH@���n{@c�
A��C�H�                                    BxZ(�  T          @�  @����r�\@K�A�\)C���@�����G�?�Q�A|��C�H�                                    BxZ)�  "          @�=q@����(�@Tz�A�Q�C���@����S33@{A�{C��f                                    BxZ)\  T          @�=q@޸R����?���A  C�q@޸R��(�?(��@�\)C�3                                    BxZ)&  T          @�\@��
��׿@  ��{C��@��
��Q쿰���.�RC�q                                    BxZ)4�  
�          @�z�@�
=@�@L(�A��A�z�@�
=?���@j�HA�\)A&�H                                    BxZ)CN  �          @��
@��>��R?�z�A?
=@(Q�@����?�Q�AB�HC�|)                                    BxZ)Q�  	�          @��@ə��+���  �E��C�y�@ə����{����C��                                    BxZ)`�  
�          @�Q�@�ff�Fff>aG�@33C�.@�ff�A녿0�����HC�u�                                    BxZ)o@  	`          @�z�@����Q�@��B�C��)@���6ff@c�
A�Q�C�e                                    BxZ)}�  
�          @�{@n{�
=@���Bd{C�z�@n{�G�@�(�BK�
C�L�                                    BxZ)��  T          @�\)@o\)�#�
@�Ba(�C�%@o\)��@��BH�C�K�                                    BxZ)�2  �          @�\@�����@���BB�C��H@���(��@��RB(p�C��H                                    BxZ)��  �          @���@��
���@���B&��C�>�@��
��@�z�B��C��R                                    BxZ)�~  �          @��H@�\)�'
=@P��A�=qC�<)@�\)�\��@A��C��                                    BxZ)�$  
�          @�z�@ƸR��@QG�A�ffC��@ƸR�I��@��A���C��                                    BxZ)��  "          @�R@�  ���?���Ag�C�AH@�  �5�?��A�C��H                                    BxZ)�p  
�          @�@���%?�33An�HC�]q@���B�\?�ffA�RC���                                    BxZ)�  "          @���@�  �K�?�33A/�
C��
@�  �\��>���@EC�H                                    BxZ*�  "          @陚@��s33>�ff@e�C�4{@��qG��333���C�T{                                    BxZ*b  �          @�(�@�p��B�\?+�@��C���@�p��G
=��=q��
C��H                                    BxZ*  "          @��@Ϯ�Mp��L�;�
=C���@Ϯ�C33��G�� (�C�ff                                    BxZ*-�  T          @�=q@�z����>�\)@{C��f@�z�������
ffC��                                    BxZ*<T  
�          @�ff@�  �Q�@\)A�\)C���@�  �AG�?�AW�C�q                                    BxZ*J�  T          @陚@Å�^�R?�Q�A8��C�+�@Å�p  >���@(Q�C�:�                                    BxZ*Y�  T          @陚@�  ��33?�(�AZ�HC�AH@�  ��>��@P  C�4{                                    BxZ*hF  �          @�Q�@���|(�?�z�A4  C�#�@����>.{?��C�XR                                    BxZ*v�  �          @�{@��
�u�?��A3
=C�s3@��
��=q>.{?��C��                                    BxZ*��  �          @��
@��H�5�@
=qA�Q�C���@��H�Vff?��HAp�C��q                                    BxZ*�8  
�          @�@����fff?�A
=C���@����q�<�>k�C��                                    BxZ*��  �          @�R@���~�R?��@��C�4{@���}p��.{��z�C�E                                    BxZ*��  
�          @�@��H�w���Q�5C�Ǯ@��H�j=q��  �   C�}q                                    BxZ*�*  T          @�Q�@�
=��G������C���@�
=�l(����H�[�C�                                      BxZ*��  �          @��@����=q>Ǯ@L(�C�U�@����
=�����C���                                    BxZ*�v  T          @�{@��\��{?c�
@�33C�O\@��\����z���z�C�*=                                    BxZ*�  �          @�33@�(���Q�?�ffAl��C���@�(����\>�=q@p�C��                                    BxZ*��              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZ+	h              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZ+              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZ+&�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZ+5Z              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZ+D               ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZ+R�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZ+aL              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZ+o�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZ+~�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZ+�>              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZ+��              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZ+��              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZ+�0              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZ+��              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZ+�|              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZ+�"              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZ+��              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZ,n              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZ,              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZ,�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZ,.`              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZ,=              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZ,K�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZ,ZR              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZ,h�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZ,w�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZ,�D              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZ,��              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZ,��              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZ,�6              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZ,��              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZ,ς              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZ,�(              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZ,��              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZ,�t              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZ-
              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZ-�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZ-'f              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZ-6              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZ-D�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZ-SX              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZ-a�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZ-p�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZ-J              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZ-��              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZ-��              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZ-�<              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZ-��              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZ-Ȉ              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZ-�.              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZ-��              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZ-�z              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZ.               ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZ.�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZ. l              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZ./              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZ.=�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZ.L^              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZ.[              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZ.i�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZ.xP              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZ.��              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZ.��              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZ.�B              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZ.��              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZ.��              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZ.�4              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZ.��              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZ.�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZ.�&              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZ/
�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZ/r              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZ/(              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZ/6�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZ/Ed              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZ/T
              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZ/b�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZ/qV              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZ/�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZ/��              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZ/�H              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZ/��              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZ/��              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZ/�:              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZ/��              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZ/�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZ/�,              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxZ0�  ,          @��@��
��@��BB��C���@��
����@��B(�C�`                                     BxZ0x  	|          @�  @�p��aG�@{�B��C�4{@�p���33@$z�A�G�C�s3                                    BxZ0!  	�          @��R@��R��  ��=q�@��C��@��R��{�Q���33C�J=                                    BxZ0/�  
�          @��@�G���Q�?n{@޸RC���@�G���G��=p���Q�C�z�                                    BxZ0>j  T          @��@�����?���A��C�K�@����z�&ff���C��                                    BxZ0M  
�          @�\)@����33@�A�(�C���@������>Ǯ@;�C���                                    BxZ0[�  	`          @�Q�@�G���{@\��A��C�!H@�G��ȣ�?�  A2�RC�33                                    BxZ0j\  "          @�G�@Z=q���@���B	(�C���@Z=q���@  A�{C���                                    BxZ0y  
(          @�G�?���Z=q@�=qBk{C�\?����ff@�z�B#  C�xR                                    BxZ0��  
�          @��
@z��e�@���BZ��C�u�@z����H@���B�\C���                                    BxZ0�N  
�          @�\)@
=q�L��@��
Bk(�C�f@
=q���H@�\)B({C�~�                                    BxZ0��  
>          @�Q�@?\)�^{@�p�BQ
=C�\)@?\)���R@�
=B��C���                                    BxZ0��  
�          @�R?�\)�g
=@��B`��C���?�\)��z�@��B��C��                                    BxZ0�@  �          @�G�?����@���B<�
C��q?���θR@hQ�A�(�C��                                    BxZ0��  �          @�G�@   ��33@�33B&��C���@   ���@@��A��HC��                                    BxZ0ߌ  
�          @�ff?&ff����@��B@C��)?&ff�ə�@c�
A��C���                                    BxZ0�2  T          @�R�#�
���@���B,�RC��=�#�
���@AG�A�ffC�<)                                    BxZ0��  
�          @�
=@�����R@�(�B)�C���@���c33@�ffBffC��\                                    BxZ1~  
Z          @�z�@����(��@�{B.ffC���@�����{@���B33C�aH                                    BxZ1$  �          @�  ?�����@��BAz�C�޸?���=q@n{A��C�H�                                    BxZ1(�  T          @�G��#�
��G�@��B'ffC�논#�
��G�@0��A���C��\                                    BxZ17p  
�          @�
=�Y����z�@u�A�
=Cl���Y����33?���Ayp�Cp��                                    BxZ1F  "          @��ÿ��H�\)@�p�B]�RCy�3���H��G�@���B�C�)                                    BxZ1T�  
�          @�\)���
�C33@�Q�Bv�CoǮ���
��Q�@��B0�RCzs3                                    BxZ1cb  �          @�
=@J�H���
@�G�B��C���@J�H�Ǯ@Q�A�Q�C�z�                                    BxZ1r  �          @�R@W
=���H@�
=BC�\)@W
=��G�@)��A��
C��=                                    BxZ1��  
�          @��H?������\@��B0�\C��R?�����
=@J�HA�33C�                                    BxZ1�T  
�          @��
@��\�8Q�@��B2��C��@��\��
=@�(�BQ�C���                                    BxZ1��  
(          @�?�33���H@���B��HC�!H?�33�r�\@У�Bk  C�n                                    BxZ1��  
�          @�
=?����(�@�B�p�C�,�?���8��@��B��3C�Y�                                    BxZ1�F  
�          @�Q�@!G�>�z�@�  B�u�@�G�@!G��@�ffB��\C�(�                                    BxZ1��  
�          @�  @N�R�
�H@��
BjQ�C�@N�R��(�@��B5�C�                                      BxZ1ؒ  �          @�=q@,�Ϳ�ff@ƸRB���C�n@,���@��@���BW33C��{                                    BxZ1�8  �          @��
@\)���
@��
B��C���@\)��Q�@p�A���C�=q                                    BxZ1��  �          @��@b�\���
@^{A��C��@b�\��p�?��AffC��                                    BxZ2�  T          @�@o\)��z�J=q��  C�AH@o\)��  �?\)���\C�p�                                    BxZ2*  T          @��@���ڏ\��Q��%C���@�����H�$z���(�C�xR                                    BxZ2!�  
�          @��R@p  ��p�?^�R@ʏ\C�5�@p  �ڏ\��
=�&�\C�`                                     BxZ20v  �          @�
=@P����p�?Tz�@�\)C�>�@P����G��Ǯ�5G�C�n                                    BxZ2?  
(          A ��@XQ���?��@�{C��R@XQ����
��{���C��                                    BxZ2M�  �          A ��@(Q���>�(�@G
=C���@(Q������33�mp�C��                                    BxZ2\h  �          A   @Q����
�����C���@Q����
�.�R����C�=q                                    BxZ2k  T          @�{?�=q��{�Ǯ�5C���?�=q��z��8Q���z�C�/\                                    BxZ2y�  T          @�p�?�R��=q�@  ���C�C�?�R�����P���Ù�C�y�                                    BxZ2�Z  �          @��H?�z���녽�Q�&ffC��?�z����
�#33��z�C��H                                    BxZ2�   �          @��H@����?0��@�G�C�޸@���\)��\�P��C�{                                    BxZ2��  T          @��?����������
=C�=q?����������\�{C�1�                                    BxZ2�L  "          @��R>�{��������C�\)>�{�\���H�C���                                    BxZ2��  
�          @���@�(��Dz�@���B��C��@�(����@Dz�A��C�Ff                                    BxZ2ј  �          @��@����@uA�p�C��\@�����@
=A�z�C���                                    BxZ2�>  "          @�@�����H@P  A��C�5�@����z�?�  A8(�C��                                    BxZ2��  
�          @�  @�33��ff@Z=qAڸRC���@�33��G�?�{AH��C�u�                                    BxZ2��  	�          @�ff@ҏ\�u@  A��C��{@ҏ\���?�Ao�C��)                                    BxZ30  T          @�=q@�G�>�\)@P  A�@�
@�G��J=q@J=qA�Q�C��\                                    BxZ3�  "          @�Q�@�(���  @c33Aܣ�C��)@�(��%�@7�A�C�aH                                    BxZ3)|  
�          @��
@�\)>#�
@B�\A���?��\@�\)�W
=@;�A��C���                                    BxZ38"  T          @��\@�Q��@p  A�ffC�xR@�Q쿫�@`��A�  C�e                                    BxZ3F�  "          @��H@��>L��@��A���?޸R@����33@}p�A��HC�q                                    BxZ3Un  �          @�p�@���\)@,��A���C��f@��
�H@
=AyG�C���                                    BxZ3d  "          @�(�@ٙ��Fff@(�A��\C���@ٙ��n{?�ffA  C��                                    BxZ3r�  �          @�(�@ٙ���@P��A�G�C�}q@ٙ��A�@=qA��C���                                    BxZ3�`  �          @��
@��H�n�R@|��A�C�(�@��H���@!G�A�33C���                                    BxZ3�  �          @�p�@��H��z�@��A���C�)@��H��  @"�\A�=qC���                                    BxZ3��  �          @�(�@���\)@�Q�BC��=@������@.�RA�=qC��=                                    BxZ3�R  �          @���@��\��{@o\)A�(�C���@��\����@ ��Ar=qC�E                                    BxZ3��  
�          @�Q�@G���  @�G�BHC�c�@G��B�\@_\)BC��)                                    BxZ3ʞ  
�          @�G��   @���@�\)Ba��B�aH�   ?�Q�@��B��B���                                    BxZ3�D  T          @�(�?���@,��@�G�B�.B�� ?���>B�\@���B�(�A�R                                    BxZ3��  �          @�ff@�
@:=q@�\B}{BZ��@�
>�ff@�z�B��AG33                                    BxZ3��  T          @�ff@4z�?�\)@�{By�Bz�@4zᾣ�
@�ffB��C��\                                    BxZ46  "          @���@�G��XQ�@�ffB7�RC��@�G���
=@��BffC��{                                    BxZ4�  �          @�(�@���@�@�\)Bz�A�{@���?�R@��RB1ff@�ff                                    BxZ4"�  �          @�G�@�  @g�@�ffB{Bz�@�  ?�33@��B9�A�z�                                    BxZ41(  T          @��H@��R@�Q�@���Bz�B"�R@��R@)��@�p�B/�A�
=                                    BxZ4?�  �          @�  @�(�?�{@�=qB=qA�Q�@�(�>�\)@�B@.{                                    BxZ4Nt  "          @��H@�(����R@�33B�\C�+�@�(��9��@xQ�A�{C�W
                                    BxZ4]  �          A=q@ʏ\��\)@��RB��C���@ʏ\�*=q@��\B  C���                                    