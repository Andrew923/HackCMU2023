CDF  �   
      time          *   Conventions       ACDD-1.3, Spase v2.2.3     title         /DSCOVR Magnetometer Level 2 One Minute Averages    id        Doe_m1m_dscovr_s20230102000000_e20230102235959_p20230104013654_pub.nc   naming_authority      gov.noaa.swpc      program       DSCOVR     summary       }Interplanetary magnetic field observations collected from magnetometer on DSCOVR satellite - 1-minute average of Level 1 data      keywords      _NumericalData.ObservedRegion.Heliosphere.NearEarth, NumericalData.MeasurementType.MagneticField    keywords_vocabulary       Spase v2.2.2   
references        �; DSCOVR TIME SERIES DATA AVERAGES ALGORITHM THEORETICAL BASIS DOCUMENT, v2.4; GSE TO GSM COORDINATE TRANSFORMATION ALGORITHM THEORETICAL BASIS DOCUMENT v2.1      metadata_link         �http://www.ngdc.noaa.gov/docucomp/page?xml=NOAA/NESDIS/NGDC/STP/Space_Weather/iso/xml/satellite-systems_dscovr.xml&view=getDataView&header=none    license       Spase.Access Rights.Open   institution       NOAA   source        DSCOVR Magnetometer Level 1    platform      'Deep Space Climate Observatory (DSCOVR)    
instrument        +boom-mounted triaxial fluxgate magnetometer    history       ,DSCOVR real-time telemetry processing system   	algorithm         FDSCOVR MAGNETOMETER LEVEL 1B DATA ALGORITHM THEORETICAL BASIS DOCUMENT     algorithmVersion      B      algorithmDate         
2015-10-15     processing_level      Level 2    processing_level_description      11-minute average using Hodges-Lehmann M-estimator      date_created      2023-01-04T01:36:54.394Z   date_calibration_data_updated         2022-11-28T00:00:00.000Z   time_coverage_duration        P01D   time_coverage_start       2023-01-02T00:00:00.000Z   time_coverage_end         2023-01-02T23:59:59.000Z   time_coverage_resolution      PT1M   creator_name      Doug Biesecker     creator_type      person     creator_institution       DOC/NOAA/NWS/NCEP/SWPC     creator_email         doug.biesecker@noaa.gov    creator_url       http://www.swpc.noaa.gov/      publisher_name         National Geophysical Data Center   publisher_type        institution    publisher_institution         DOC/NOAA/NESDIS/NGDC   publisher_email       william.rowland@noaa.gov   publisher_url          http://www.ngdc.noaa.gov/dscovr/   records_maximum         �   records_present         �   records_data        '   records_fill         y   records_missing                    time                description       "date and time for each observation     
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
short_name        overall_quality    C_format      %d     units         n/a    lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale        7lBxV�r@  T          @��Ϳ�G�@�(���R����B��)��G�@�=q��{�R�HB���                                    BxV���  
�          @�{�˅@����ff��  B��f�˅@����xQ��6ffBڅ                                    BxV���  �          @����{@�z��ff�ʣ�B�k���{@�G��z�H�8(�B�                                      BxV��2  
�          @�p���@��R�޸R���B�G���@�Q�(����
B�k�                                    BxV���  T          @�p�����@�G�� ����\)B��Ὲ��@���Y���!B��
                                    BxV��~  T          @���@��
�+��33B��ü�@����ff����B��
                                    BxV��$  �          @�녿��\@w��!G�� ��B�zῂ�\@�(��������B�.                                    BxV���  "          @��ÿ�  @�{��
=��z�B�LͿ�  @�G��L���
=B��
                                    BxV��p  T          @���>#�
@�
=�	����{B��=>#�
@�(���G��E��B��                                    BxV��  T          @����(�@��H��
��RB��f��(�@�G������l  B��R                                    BxV��  
�          @�p��(��@���G���
=B�33�(��@���h���6ffBĳ3                                    BxV�b  �          @��   @�=q�:�H��B�p��   @�33>�G�@��B�W
                                    BxV�"  T          @�33����@�������s�B�녿���@��ý����
B�ff                                    BxV�0�  �          @�����@��׿xQ��F�HB��f���@�(�>\)?��HB�33                                    BxV�?T  T          @������@��\�Q��$Q�B��H����@�z�>���@n{B�aH                                    BxV�M�  "          @��Ϳ�(�@�G���Q���Q�B߳3��(�@�  ?:�HA��B��                                    BxV�\�  "          @�=q���
@��>��R@{�B۽q���
@���?�\)A��HBݮ                                    BxV�kF  �          @����{@���?uAA�B�.��{@y��@ ��A�G�B�\)                                    BxV�y�  �          @�\)��Q�@e?�33A�G�B��Ϳ�Q�@HQ�@��A�(�B�R                                    BxV���  
�          @�����H@xQ�?��\AXz�B����H@`  ?��HA��B�
=                                    BxV��8  �          @�z΅{@a�?�p�A�p�B�
=��{@:�H@333B$Q�B��)                                    BxV���  �          @�
=���H@��
>\)?�ffB����H@|��?�Aw\)B܅                                    BxV���  �          @�ff��\)@^�R����B�B���\)@~{��p���\)B��                                    BxV��*  �          @����5�@?�\)A���CY��5�?�{@
=qB�C��                                    BxV���  
�          @��\�?\)@�?��RA��
C��?\)?�G�@ ��B�C��                                    BxV��v  
�          @��K�?�(�@Q�B��CG��K�?��@333BQ�C��                                    BxV��  �          @�ff�AG�@)��?�{A�=qC
���AG�@
=@p�B��C�                                    BxV���  
�          @�{�AG�@333?ǮA�C	#��AG�@�@p�A�RCh�                                    BxV�h  
�          @�Q��9��@Mp�?�
=At��C)�9��@4z�?���AͮC޸                                    BxV�  
�          @���,(�@Y��?���A��HC W
�,(�@<��@
=qA�
=CL�                                    BxV�)�  "          @���z�@�z�?
=q@�33B�\�z�@xQ�?�ffA�G�B��                                    BxV�8Z  T          @�Q쿦ff@�G��Ǯ��{B��f��ff@�  ?@  A=qB�8R                                    BxV�G   T          @�  ��\)@��
��z����\BШ���\)@��\��\)�W�B�aH                                    BxV�U�  �          @�ff����@�Q��  ��Q�B��Ϳ���@�  ������z�B�G�                                    BxV�dL  
�          @��;�\)@�  �����B�����\)@��R���
�r�\B��f                                    BxV�r�  �          @�\)>�\)@g��Tz��)B�Q�>�\)@��\�z���33B��q                                    BxV���  "          @�{�&ff@g��N{�%
=B�=q�&ff@�����R��
=B�{                                    BxV��>  �          @��R�&ff@�Q���R��G�B�B��&ff@���{�T(�BøR                                    BxV���  "          @���<#�
@�����
���B��<#�
@�p��aG��!B��3                                    BxV���  �          @���?\(�@|(��B�\�=qB�W
?\(�@�녿�p�����B���                                    BxV��0  �          @�33@�@l(��&ff��33Bt�\@�@�
=��{��B��                                     BxV���  
�          @�G�?�z�@��׿����{B��?�z�@�=q�!G���ffB��)                                    BxV��|  �          @��
?�p�@R�\�g
=�4  B��{?�p�@�=q�,(���z�B��                                    BxV��"  T          @���@   @�H���\�Q��BI��@   @Vff�W
=�#  Bl��                                    BxV���  �          @��@G�@������V�BB  @G�@N�R�^{�)=qBg��                                    BxV�n  �          @��?�(�?�\)��=q�g\)B-�R?�(�@7��n�R�<  B]�H                                    BxV�  8          @��=�G�@n{��\���\B���=�G�@�p�������B���                                    BxV�"�            @��H?��H@*=q���R�c{B�?��H@j�H�k��.\)B�p�                                    BxV�1`  "          @�  ?Ǯ?��������fB%�\?Ǯ@   ��G��^33Bh                                      BxV�@  �          @�  ?�=q@�R��=q�k�RBl��?�=q@N{�h���933B���                                    BxV�N�  T          @�{?���@H���s�
�@=qB��{?���@~{�<(��B��R                                    BxV�]R  
(          @�  ?�z�@@����G��J33B��?�z�@y���L(��(�B�W
                                    BxV�k�  �          @�@�@Z�H�R�\�Q�Bm  @�@��
�
=��B~                                    BxV�z�  �          @�{@�@S�
�W��"Q�Be�\@�@����{��p�By
=                                    BxV��D  
�          @�(�?�@z����\�g��Bj(�?�@S33�h���5B�p�                                    BxV���  T          @�(�?��@ff�����k33BV��?��@E��j�H�;G�B}��                                    BxV���  �          @��Ϳz�H@��Ϳ�����B�  �z�H@��\)��
=B˔{                                    BxV��6  �          @�ff���H@�p��33�ϮB�W
���H@�G��z�H�AB�B�                                    BxV���  
�          @�p�>��H@j=q�N�R�$�
B��f>��H@��\������B�8R                                    BxV�҂  "          @���?:�H@w
=�J�H�Q�B��?:�H@�Q�������B��                                    BxV��(  �          @�(�>�{@|(��*�H�{B�33>�{@��R��33���B�W
                                    BxV���  
�          @���{@u�?#�
Az�B�.�{@c�
?��A�p�B���                                    BxV��t  "          @�  �'
=@b�\?0��A\)B����'
=@QG�?\A��C �
                                    BxW   �          @�z��K�@(Q�?�  A�=qCh��K�@��?�A�C�f                                    BxW �  �          @��H��G�@x�ÿO\)�/�Bހ ��G�@~{>\)?�z�Bݳ3                                    BxW *f  
�          @��;�=q@c�
�:�H�=qB��R��=q@��Ϳ��R�̏\B�z�                                    BxW 9  �          @��H���\@B�\���
���HB�{���\@W���G���
Bՙ�                                    BxW G�  �          @���U�@L(�>\@�\)C5��U�@@��?���AaC	��                                    BxW VX  T          @�(��N{@R�\>�33@�=qCff�N{@G
=?�\)A`  C�R                                    BxW d�  �          @��7�@j=q�8Q����C {�7�@fff?0��A  C �=                                    BxW s�  �          @�33�G�@o\)��������B��G�@|�;�
=��=qB�q                                    BxW �J  "          @��   @b�\�
�H��ffB��f�   @|�Ϳ��\��  B��                                    BxW ��  �          @�
=����@'
=���fBŀ ����@a��\���0�RB�                                    BxW ��  
Z          @�zῳ33@s�
�!G���Q�B�G���33@�����ff��p�B�
=                                    BxW �<  T          @�p���@q��!����\B���@��׿�=q��B�q                                    BxW ��  
�          @��R��ff@o\)��Q����
B�\)��ff@��H��G��V=qB���                                    BxW ˈ  
�          @��\?�
=@a��#�
�
�B��?�
=@��ÿ�z����HB�p�                                    BxW �.  �          @��R�,��@E���R���C0��,��@H��=�?�C�f                                    BxW ��  "          @���J=q@0  ���
���
C
��J=q@0  >���@�Q�C
�3                                    BxW �z  
�          @�Q��Mp�@.{�\)���
C���Mp�@1�=�Q�?�z�C)                                    BxW   T          @�G��0��@G
=>�p�@���C���0��@<(�?��AqC&f                                    BxW�  T          @����R@g
=?&ffA
ffB�z���R@W
=?��HA��B�.                                    BxW#l  
�          @��Ϳ��
@���?\)@��Bݙ����
@r�\?�p�A��B���                                    BxW2  "          @��R�   @Q녿B�\�&=qB����   @W�<��
>���B�8R                                    BxW@�  �          @����ff@^�R��R�\)B�#��ff@a�>L��@(Q�B�aH                                    BxWO^  T          @����2�\@H�ÿ�\)��z�C���2�\@[��Y���.=qC�                                    BxW^  �          @��\�AG�@AG��У�����C
=�AG�@S�
�fff�5�C^�                                    BxWl�  T          @�Q��P  @7
=��Q��v�HC
�{�P  @C�
��\�θRC��                                    BxW{P  
�          @����%@j=q�u�A�B����%@g�?(�@���B�B�                                    BxW��  �          @�Q��Q�@q논��
�uB�aH�Q�@k�?W
=A.�\B��R                                    BxW��  �          @�ff�   @e?:�HA33B����   @U�?��
A�G�B��q                                    BxW�B  �          @����z�@u��
=���B� �z�@w
=>�z�@n{B�
=                                    BxW��  k          @�Q��
=q@p�׿��H�yp�B����
=q@{����
��(�B�\                                    BxWĎ  o          @�33�+�@�?�A�  C
��+�?���?�z�AC�                                    BxW�4  �          @�
=�7�@&ff@	��A��C	� �7�@33@+�B  Ck�                                    BxW��  �          @�{�^�R@��?��\A�CxR�^�R@33?��
A�{C��                                    BxW��  �          @�  �,(�@�@-p�B��C��,(�?�z�@J�HB3��CW
                                    BxW�&  �          @����*=q@<(�@	��A�C��*=q@Q�@0��B�C
!H                                    BxW�  �          @���4z�@Fff?��
A���CL��4z�@+�@(�A�
=C�                                     BxWr  "          @��׿��H@\)>�
=@��
B�aH���H@r�\?�ffA���B�3                                    BxW+  T          @��׿Ǯ@�z�?=p�AQ�B�W
�Ǯ@w�?�33A�z�B���                                    BxW9�  �          @�  ��
=@�z�?^�RA3�B�  ��
=@u?�\A���BܸR                                    BxWHd  "          @��ÿ�@��H?(�@�Q�B�R��@vff?�G�A��
B�Q�                                    BxWW
  
�          @�Q���@��?=p�A\)B�p����@vff?У�A���B�#�                                    BxWe�  T          @�  �˅@��H?k�A>{Bފ=�˅@q�?�A�{B��                                    BxWtV  �          @�{����@~�R>�{@��RB�{����@s33?��HA�ffB�\                                    BxW��  
�          @�{��@^�R?���Ah(�B�����@I��?�A�z�B�8R                                    BxW��  T          @�\)���
@w�?^�RA8��B�k����
@e?ٙ�A�Q�B��
                                    BxW�H  �          @��\��G�@|(�>Ǯ@���B�녿�G�@p��?�  A���B�                                    BxW��  �          @�G��AG�@>�R?��
A�z�C^��AG�@%�@	��A��C�\                                    BxW��  "          @�����H@h��?xQ�AG\)B�33��H@U?޸RA��HB���                                    BxW�:  T          @�
=�Z�H@&ff?^�RA7
=CǮ�Z�H@ff?�A��RC�\                                    BxW��  �          @���o\)?�{?��
A��HC�
�o\)?��
?�z�A��\C�R                                    BxW�  �          @���xQ�>��?��AθRC0&f�xQ���?�33A�{C6G�                                    BxW�,  "          @�ff�w
=?�
=?J=qA*{C�\�w
=?��H?���Au��C=q                                    BxW�  "          @�ff�c33@%>��H@�(�C޸�c33@�H?��AY��C��                                    BxWx  �          @��
�`  ?���?У�A���C�
�`  ?�ff@G�A�Q�C\                                    BxW$  
Z          @�z��hQ�@�?
=@��CG��hQ�@	��?���Ad��CY�                                    BxW2�  �          @�z��Vff@2�\>u@L��CJ=�Vff@*�H?W
=A0��C�                                     BxWAj  T          @�z��P��@0  �k��EC޸�P��@/\)>�Q�@�(�C                                    BxWP  T          @���Z�H@0��    ��C��Z�H@,��?
=@�C�R                                    BxW^�  "          @�ff�Q�@<(�����  C
��Q�@>�R=�?�G�C	�3                                    BxWm\  �          @�  �P  @@  �E��=qC	@ �P  @Fff����(�C\)                                    BxW|  "          @�\)�P��@C33�u�FffC���P��@A�>��@�Q�C	!H                                    BxW��  �          @��R�]p�@(�þW
=�333C�f�]p�@'�>�33@��
C��                                    BxW�N  T          @��
�w�?�{?
=q@�
=CJ=�w�?��H?k�AEG�C33                                    BxW��  �          @�{�hQ�@"�\�k��<��C���hQ�@!�>��R@�G�C\                                    BxW��  �          @�(��k�@�>�z�@q�C�H�k�@{?E�A"�RC��                                    BxW�@  �          @��H�i��@G�?k�AF�\C���i��?��?���A�(�C��                                    BxW��  �          @�33�W�@�R?���Al��C���W�@(�?���A�p�C��                                    BxW�  �          @���^{@%�>�@ÅCaH�^{@�H?}p�AR�\C
=                                    BxW�2  �          @��\�s�
?�=q��G���  C^��s�
?�׽L�Ϳ=p�C��                                    BxW��  �          @����k�?��?�ffAfffC)�k�?��
?�33A��\Ck�                                    BxW~  "          @����qG�?��?
=A Q�Cc��qG�?�(�?xQ�AR�HCh�                                    BxW$  �          @���XQ�@  ?W
=A<  Ck��XQ�@G�?��A��HC)                                    BxW+�  "          @�Q��  @,��?��A�(�C���  @��@=qB33C�                                    BxW:p  �          @���`  ?�Q�?�ffAr�\C:��`  ?�Q�?���A�  C��                                    BxWI  �          @��
�i��@�>W
=@5C
=�i��@(�?+�A(�C
                                    BxWW�  "          @��
�'
=@-p�?��RA�33C���'
=@  @   B  CO\                                    BxWfb  �          @�{���@)��@1G�BG�C�����@�@P  B=p�C	L�                                    BxWu  T          @�33�L��?��@%B�C��L��?=p�@333B!��C'                                    BxW��  �          @��
�Z�H?\(�@#�
B{C%�f�Z�H>��
@+�B  C.��                                    BxW�T  �          @�33�/\)=u@VffBJ�HC2��/\)�(�@R�\BFG�C@��                                    BxW��  �          @���A�>�ff@EB4��C+���A녾8Q�@G�B6�HC7T{                                    BxW��  �          @��\�N{>�z�@6ffB%z�C.޸�N{��z�@6ffB%z�C9
                                    BxW�F  �          @��
�\��>u@'�B�C0
�\�;�z�@'
=B�C8�{                                    BxW��  �          @��R�vff?��?��A��C �)�vff?z�H?�\)AɮC%��                                    BxWے  T          @���{�@33=�Q�?���Cu��{�?��R>��H@�33C!H                                    BxW�8  T          @�Q��|(�@Q�#�
�.{C�=�|(�@>��@��C�q                                    BxW��  
�          @�\)�~�R@   >�Q�@��CG��~�R?��?@  A�RC��                                    BxW�  
�          @�
=�s33@�?k�A?
=CǮ�s33?�33?���A�{Cff                                    BxW*  �          @�  �p  @p�?p��AC�Cn�p  ?��R?�{A�(�C{                                    BxW$�  �          @�Q��w�@
=?c�
A6{Cc��w�?�33?��
A��\C��                                    BxW3v  T          @���tz�@�?aG�A6�HC���tz�?�z�?��
A�p�Cu�                                    BxWB  �          @�Q����
?�=q=u?Q�C&����
?�ff>�=q@^�RC&z�                                    BxWP�  �          @����~�R@z�>\@�33C���~�R?��H?E�AffC�
                                    BxW_h  �          @�\)�u@�<#�
>B�\CQ��u@�R>�ff@��HC�{                                    BxWn  T          @�  �y��@
�H�k��<��C�3�y��@
�H>B�\@��C��                                    BxW|�  T          @�  �~{@��u�@��Cc��~{@�>.{@Q�CQ�                                    BxW�Z  T          @����n�R@=q�=p��(�C)�n�R@ �׾�=q�[�C\                                    BxW�   �          @���vff@
=�Q��)�C@ �vff@�R������RC�                                    BxW��  �          @���tz�?��Ϳ����C)�tz�@z�u�G\)C��                                    BxW�L  "          @����W
=@�?�Q�A�C{�W
=@
=?��AʸRC�=                                    BxW��  �          @�(��Dz�@��@  A�(�C{�Dz�?�
=@'
=B{CL�                                    BxWԘ  �          @��Y��?�Q�?�AУ�CY��Y��?��@  A�(�C�)                                    BxW�>  �          @�  �hQ�@"�\?�R@�C
=�hQ�@Q�?�=qA_�
C�                                    BxW��  �          @�\)�h��@p�>�\)@l��C�3�h��@�?=p�A=qC�                                    BxW �  �          @�ff�w
=@
=>�G�@�Q�CJ=�w
=?��R?Tz�A,��C��                                    BxW0  T          @���s33?��?���A���C�3�s33?��?�p�A�  C �
                                    BxW�  T          @�{���?�(�?�G�ATQ�C#n���?�G�?�(�A��\C&�                                    BxW,|  �          @�
=����?���=u?L��C$xR����?�z�>�\)@g
=C$޸                                    BxW;"  �          @�Q����?�{��33��
=C%�����?�z�\)��\C$�3                                    BxWI�  �          @�Q��aG�@'
=��\)�k�Cs3�aG�@'�>L��@'
=C^�                                    BxWXn  �          @��R�tz�@
=q�8Q��z�Cu��tz�@�׾��R����Cff                                    BxWg  �          @�ff�x��?�=q�p���E�Cٚ�x��?�(��
=��\)C.                                    BxWu�  �          @��R�vff?�녿�
=��33C�H�vff?�׿�{�i�C�                                    BxW�`  �          @�Q��y��?�\��G���33C���y��?�(��k��>�RC@                                     BxW�  
�          @��y��?������
�k�Cn�y��?�>���@��
C�                                    BxW��  �          @�z��E?�{@)��B=qCc��E?�{@9��B%�
C &f                                    BxW�R  �          @���aG�?�33@=qBG�C!���aG�?333@$z�B�RC(��                                    BxW��  �          @���R�\?�\)@#33B=qC�{�R�\?��@2�\B  C ��                                    BxW͞  �          @�ff�P  ?�z�@/\)B�RC L��P  ?(��@9��B$�RC(}q                                    BxW�D  �          @�ff�1�?=p�@L��B@�C%  �1�>.{@Q�BF�C0}q                                    BxW��  �          @��\��?�G�@k�BZ��C	�Ϳ�?��@{�Br�C�                                     BxW��  �          @�����
@   @W
=B>�C+���
?���@j=qBV�\C&f                                    BxW6  T          @��\�
=?��H@aG�BK
=C	+��
=?��@s�
Bc�C
                                    BxW�  �          @���{@z�@W�B9�C�R�{?ٙ�@n{BTffC�                                    BxW%�  �          @�p���z�@�@^�RBCffC �)��z�?�p�@u�B`p�C	                                    BxW4(  �          @�33�%@5@\)B�RCh��%@�@<(�B�RC	�
                                    BxWB�  �          @�33�
=q?�{@l��BW��C0��
=q?u@z�HBk�HC\                                    BxWQt  T          @��׿��H?�  @~�RBw�
C����H?��@�(�B��C":�                                    BxW`  �          @�����p�?��
@j�HBX=qC
\��p�?���@z�HBoG�CE                                    BxWn�  �          @�=q�(�@p�@H��B.�HC	�\�(�?�33@^{BF��C��                                    BxW}f  �          @��\�'
=@3�
@��B(�C�{�'
=@
=@8��Bp�C	�
                                    BxW�  �          @�33��33@;�@.{Bz�B�
=��33@�@J�HB6�C 
=                                    BxW��  �          @�
=����@HQ�@P  B0=qB��f����@#33@n{BS��B�k�                                    BxW�X  T          @����@e@$z�B=qB�p����@G�@HQ�B(  B螸                                    BxW��  �          @�zῧ�@o\)@�A�  Bڅ���@S33@:�HB�B�(�                                    BxWƤ  �          @�����@qG�@A�G�B�aH���@W�@+�Bz�B�                                      BxW�J  �          @�33�Q�@���?�(�A�G�B�녿Q�@i��@'
=B�B�8R                                    BxW��  �          @�p���ff@�ff?�
=A�p�B�
=��ff@w�@A�(�B�Q�                                    BxW�  �          @��׿��\@���?Tz�A"{B�\)���\@��?ǮA��B�u�                                    BxW<  �          @�33��G�@�G�?:�HA
ffB��H��G�@��?���A�B�G�                                    BxW�  T          @���
=q@���?\)@��HB�ff�
=q@��?��RAmp�B�                                      BxW�  �          @�33��z�@��?\)@�p�B��ÿ�z�@�Q�?�G�AqB�ff                                    BxW-.  �          @�=q��\)@�(�?(�@�\B�=q��\)@�
=?�ffA|  B�R                                    BxW;�  �          @���>#�
@{�������RB�Q�>#�
@��H�(���\)B��                                     BxWJz  T          @��@(�@]p��+���Bf�\@(�@u�ff����Bq�                                    BxWY   �          @��R@@\���5��Bk�@@w
=�G��ڏ\Bv{                                    BxWg�  T          @��@\)@P  �>{���B]z�@\)@j�H����p�Bj\)                                    BxWvl  �          @�@�
@C33�Dz���
BS�@�
@_\)�#33��
=Bb{                                    BxW�  T          @���@:�H@#�
�L(��ffB$��@:�H@AG��0  ���B7�                                    BxW��  �          @��@'
=@�R�e�9Q�B"  @'
=@0  �L��� ��B:G�                                    BxW�^  �          @��@�R@z��o\)�?(�B,@�R@7��U�%��BD�                                    BxW�  �          @��@+�@4z��N{��\B9�H@+�@Q��0  �(�BK
=                                    BxW��  �          @�Q�@!G�@O\)�6ff�(�BPz�@!G�@h�������
B]{                                    BxW�P  �          @��R?��R@�������\)B�aH?��R@�z�(����B��                                    BxW��  �          @�Q�@.�R@6ff�>{�  B8�@.�R@P��� ����Q�BHp�                                    BxW�  �          @��@i��?��\�QG��#  Ax��@i��?�  �E����A�{                                    BxW�B  �          @�Q�@8Q�?�p��Z�H�>�A���@8Q�?�(��L���.z�A�G�                                    BxW	�  �          @�33@	��@S�
�0  �p�Bc��@	��@k���R����Bn�                                    BxW	�  �          @�33?���@��������[
=B�B�?���@�p������B��f                                    BxW	&4  �          @���?��@��H��ff�j�\B��{?��@�\)�z���{B�G�                                    BxW	4�  �          @�ff?\(�@��H����=qB���?\(�@��ÿs33�((�B��=                                    BxW	C�  T          @��R?=p�@��\���H���\B���?=p�@��׿}p��0��B�W
                                    BxW	R&  �          @��?�\@��׿h���*�\B�(�?�\@�33��  �:�HB��H                                    BxW	`�  �          @��?h��@�
=�	���ӮB��?h��@�\)�\����B�{                                    BxW	or  �          @�{?+�@�p�����\)B��H?+�@�p�����Q�B�Ǯ                                    BxW	~  T          @�\)?z�@�Q��p����RB�ff?z�@�������w�B��                                    BxW	��  "          @���?�{@�Q�B�\���B���?�{@�=q���
�^�RB���                                    BxW	�d  T          @�z�?n{@�p�?��AHz�B�aH?n{@�\)?��HA���B�k�                                    BxW	�
  �          @�33>�@�(���  ���B��\>�@����^�R�,  B�                                      BxW	��  T          @���h��@z=q@G�A�p�B�(��h��@c�
@2�\B��BШ�                                    BxW	�V  �          @�(���  @��\?z�@�RB�G���  @�ff?��HArffB�u�                                    BxW	��  "          @�Q�>L��@�(��aG��,Q�B�\>L��@��R�u�7�B�#�                                    BxW	�  "          @���?8Q�@��\�p���6�RB�?8Q�@�p������i��B�\)                                    BxW	�H  �          @�33?�G�@��
���
�D  B�=q?�G�@��R�Ǯ���B��q                                    BxW
�  T          @���?
=@������\)B�Ǯ?
=@�����
�G\)B�\)                                    BxW
�  T          @��?h��@��Ϳ�ff���\B��q?h��@�33��Q��d  B��q                                    BxW
:  "          @��?�  @��ÿ����J�RB�G�?�  @�(������
B�
=                                    BxW
-�  "          @�(�?�@��R����
=B�\)?�@�zῇ��L��B��f                                    BxW
<�  
�          @��>��H@�{��{��33B�ff>��H@��Ϳ�G��q��B���                                    BxW
K,  
�          @��>���@��H�s33�;33B��q>���@�p���{��B��H                                    BxW
Y�  
�          @�
=���R@�{>L��@&ffB����R@��
?G�A!G�B�#�                                    BxW
hx  T          @��?xQ�@�=q�
=��B�G�?xQ�@��H����\)B��f                                    BxW
w  �          @���?c�
@n�R�6ff��B���?c�
@�����G�B�#�                                    BxW
��  �          @�ff?L��@qG��,(���\B�  ?L��@��\����p�B���                                    BxW
�j  T          @�ff?J=q@h���7��z�B�p�?J=q@~{�Q���B�u�                                    BxW
�  "          @�
=?u@o\)�,(���\B�W
?u@�������ޣ�B�u�                                    BxW
��  "          @��?�@x���&ff��HB���?�@�{���z�B��)                                    BxW
�\  �          @��
>�
=@fff�6ff��
B�>�
=@{��Q�����B��)                                    BxW
�  �          @�
=?z�@vff�(�� =qB��?z�@�(���Q���  B�
=                                    BxW
ݨ  T          @�ff?���@9���j=q�C�\B�#�?���@U��Q��)�\B��                                    BxW
�N  "          @��?��?�������qG�BK33?��@����  �Zp�Be(�                                    BxW
��  
�          @��?�ff?����z��tG�B]��?�ff@Q��xQ��\ffBu\)                                    BxW	�  �          @�{?G�@  ��  �m
=B��R?G�@.{�l���R�B��                                    BxW@  �          @��?!G�@K��R�\�5�
B��R?!G�@c33�8����RB�                                      BxW&�  T          @�{���@fff� ����B�(����@w��z��߮B�=q                                    BxW5�  
�          @�녿c�
@.{�I���>�B�\�c�
@E��3�
�$�HB��                                    BxWD2  
�          @����Q�@XQ����ϮB�\)�Q�@Z�H��\)�uB��
                                    BxWR�  �          @����Tz�?޸R?��A�Q�CO\�Tz�?\?�p�A�  Cc�                                    BxWa~  T          @���:=q@ ��?�33Aң�C#��:=q@�@�A�\C�                                    BxWp$  T          @�
=�?\)@p�?���Aݙ�Cu��?\)?�(�@��A�  C��                                    BxW~�  T          @�\)�=p�@6ff�!G��33C��=p�@9���������
C��                                    BxW�p  �          @���J=q@��ff��p�C=q�J=q@�׿�����{Cp�                                    BxW�  "          @��\�
=@HQ쾸Q���=qB���
=@I���#�
��B��\                                    BxW��  
�          @��
���@�33?&ffA
ffB����@�  ?�{Alz�B���                                    BxW�b  T          @��\���@n�R>��R@��B����@j�H?8Q�Ap�B��                                    BxW�  �          @�\)�
=q@i������\B�8R�
=q@h��>�=q@k�B�Q�                                    BxW֮  T          @�=q�(��@Dz�B�\�+�C���(��@Dz�>��@C��                                    BxW�T  �          @�  ����?��R�W
=�d�\C}q����?����J�H�RffC^�                                    BxW��  T          @������?�  �n{G�C.����?���dz��s{B��                                    BxW�  �          @�33��G�?}p��}p��C����G�?�z��u��v��C�                                    BxWF  �          @��R��
=?���vff�p�CE��
=?��H�n{�c��C޸                                    BxW�  �          @�33�6ff?���!��p�C�H�6ff@Q��z��z�C.                                    BxW.�  �          @���AG�?���=q���C  �AG�@Q�������\C�\                                    BxW=8  "          @��
�Z=q?�����
� �C  �Z=q?�
=�	����Q�C�                                    BxWK�  �          @��R�S�
?���)����C� �S�
?�ff�   �	��C�)                                    BxWZ�  �          @�
=�Tz�?��H�����C�R�Tz�?��������C��                                    BxWi*  "          @��N�R?�=q�G���33Cz��N�R@33�z���33C�)                                    BxWw�  �          @�z��\(�?����
�H��RCxR�\(�?��
�   ��z�C�H                                    BxW�v  "          @��`  ?�  �\)��C�
�`  ?��H�����\C�                                    BxW�  �          @���Vff?����33���Cٚ�Vff@�ÿ�����p�Cp�                                    BxW��  T          @�(��Tz�?޸R������HCc��Tz�?����G���
=C��                                    BxW�h  �          @�Q��N�R?�  �8���ffC���N�R?\�0  ���C�3                                    BxW�  �          @��R�U�?��\�,���\)C&f�U�?\�#�
���Cn                                    BxWϴ  �          @�33�b�\@�?aG�A<��C.�b�\@(�?��Al  CG�                                    BxW�Z  �          @��\�Z=q@  ?s33AR{C�
�Z=q@	��?�z�A��RCǮ                                    BxW�   T          @���e?�׿�(���CaH�e?�p����
�b�\C\                                    BxW��  
�          @����mp�?�녿�(����C33�mp�?޸R����g�
C�
                                    BxW
L  �          @��`��@�
�����C���`��@�W
=�>�RC33                                    BxW�  
Z          @�(��xQ�?�ff�
=q���RC$�H�xQ�?����ff���C$B�                                    BxW'�  "          @�ff�\)?s33�W
=�>�RC&�f�\)?u����\C&p�                                    BxW6>  
�          @�������>����J=q�1p�C/������>�p��B�\�*�\C.�                                    BxWD�  �          @���l(�?8Q��p���\)C(��l(�?^�R��z����RC&��                                    BxWS�  �          @��R�u�?}p�������  C%�\�u�?�{�����\)C#�                                    BxWb0  
Z          @��R��Q�>�33�����n{C/��Q�>�G�����f=qC-�R                                    BxWp�  "          @�p��p  =L�Ϳ�����33C30��p  >W
=�����
C0Ǯ                                    BxW|  T          @�
=�u>�녿����RC-�)�u?�Ϳ�\)��C+Ǯ                                    BxW�"  T          @���s�
?����
��  C,=q�s�
?&ff��p���(�C*Y�                                    BxW��  �          @����z�H?:�H��  ����C)h��z�H?\(���
=���C'��                                    BxW�n  "          @����xQ�>���G���C-0��xQ�?(����H���\C+
                                    BxW�  "          @�
=��G�?���h���H��C+����G�?+��\(��=G�C*�=                                    BxWȺ  "          @�����?J=q�W
=�7�C(�3���?\(��E��(��C(�                                    BxW�`  �          @��R��=q?@  �+���RC)�{��=q?L�Ϳ�����C(�H                                    BxW�  "          @����H?
=q�(��33C,h����H?
=����=qC+                                    BxW��  �          @����33?
=�\)��p�C+�{��33?�����Ϳ�=qC+�3                                    BxWR  
�          @��
���H�aG�=�G�?�G�C7����H�k�=�Q�?��\C7.                                    BxW�  T          @�z���=q��{>�
=@�C8����=q�\>���@�G�C9E                                    BxW �  �          @�����
��Q쾙����p�C9����
��{��������C8�R                                    BxW/D  
Z          @�����(��L�;������HC6����(��.{��33��G�C6aH                                    BxW=�  
�          @�����\=�G��8Q�� ��C2c����\>.{�5�=qC1�H                                    BxWL�  �          @��R�}p�?녿����33C+���}p�?(�ÿ��R���C*z�                                    BxW[6  T          @��}p�?n{�@  �'�C&�R�}p�?}p��.{��C&                                      BxWi�  �          @�����?��ý��Ϳ�
=C%�����?��ü���33C%
=                                    BxWx�  "          @�ff��33?c�
����z�C'�q��33?fff��Q쿘Q�C'�H                                    BxW�(  
�          @��
��  ?u>�?�{C&n��  ?s33>L��@3�
C&�
                                    BxW��  "          @��z�H?�\)=�?��HC ���z�H?�{>aG�@B�\C �)                                    BxW�t  T          @���|��?�  <#�
=�Q�C"c��|��?�  =�Q�?�ffC"p�                                    BxW�  �          @�G��h��?��
?O\)A<��C �f�h��?�(�?c�
AQp�C!xR                                    BxW��  
�          @��H�'�?�{@   B��C�=�'�?�Q�@'�B \)C&f                                    BxW�f  "          @����P  ?\)@	��BffC*J=�P  >��@�B��C,�=                                    BxW�  
Z          @��@  ?333@.{B&{C&�)�@  ?�@0��B)�C*33                                    BxW��  T          @�  ���@(�@I��BA33B�Q쿱�@�R@S33BN
=B��H                                    BxW�X  
�          @�\)�p�?��
@?\)B:{C#��p�?�=q@FffBC
=C^�                                    BxW
�  T          @���:=q?�@'
=B\)C
=�:=q?�  @,(�B!p�C�=                                    BxW�  
�          @�����@L(�?�\)A}�B����@G�?��A�Q�B�p�                                    BxW(J  T          @�녿�{@�G�?=p�A�
BҞ���{@\)?}p�AV�\B�                                    BxW6�  �          @�녿��\@�=q=L��?=p�B֞����\@��>���@��\Bֳ3                                    BxWE�  
�          @��H��{@���?�@�(�B�G���{@\)?E�A&�\Bٞ�                                    BxWT<  T          @����u@aG�@(�A��Bң׿u@XQ�@��B�B���                                    BxWb�  �          @�����G�@]p�@{B(�B£׾�G�@S�
@*�HB�B�L�                                    BxWq�  T          @�녾W
=@j�H@{A��B��׾W
=@b�\@�B	��B��H                                    BxW�.  T          @�G��aG�@r�\?�G�A�z�B�.�aG�@l(�?�(�A��B��                                    BxW��  T          @�녿�ff@j=q?޸RA�p�B��ῦff@c�
?���AٮB�
=                                    BxW�z  T          @�녿�=q@Q�@A���B�Q��=q@J=q@G�A��B�(�                                    BxW�   �          @�G�����@Mp�@z�A��B�p�����@E@\)A�  B�W
                                    BxW��  �          @�G���\)@XQ�?�33A�{B��ÿ�\)@QG�@�A癚B�=                                    BxW�l  "          @�p��C�
?�@
=A�p�CY��C�
?ٙ�@��B C�R                                    BxW�  "          @�p��dz�?!G�?�
=A��
C*  �dz�?�?��HA�C+�{                                    BxW�  �          @���s33��?�=qA�  C:�3�s33�\)?�ffA��HC<T{                                    BxW�^  "          @�p��o\)��Q�?���A���C9���o\)��ff?�ffA�ffC:�                                    BxW  T          @�G��g����?�p�A�
=C:u��g��   ?��HA�=qC;�=                                    BxW�  T          @���c�
��?��A��HC;\)�c�
���?���A�C<�\                                    BxW!P  "          @�G��fff�Q�?У�A�z�C@�)�fff�fff?˅A�\)CB�                                    BxW/�  �          @����S�
=�@�
Bz�C1�q�S�
<#�
@�
B�C3��                                    BxW>�  T          @���hQ��?�Aޣ�C4� �hQ��?�z�A�=qC5��                                    BxWMB  
�          @�G��^{��z�@G�A��HC8�
�^{�Ǯ@   A���C:Y�                                    BxW[�  �          @����fff<�?���A�ffC3��fff�L��?���A�ffC4��                                    BxWj�  
�          @��
�w�?J=q?��AuC(z��w�?=p�?�\)A}��C).                                    BxWy4  �          @���u?0��?��Ayp�C)��u?&ff?�\)A�(�C*xR                                    BxW��  
�          @x���^�R?0��?�ffA�Q�C(Ǯ�^�R?�R?�=qA��C)�{                                    BxW��  �          @}p��h��?�33?��
Ap��C"s3�h��?�{?�=qA|Q�C#{                                    BxW�&  
�          @�  �j�H?�
=?+�A=qC���j�H?�z�?:�HA'�
C\                                    BxW��  �          @~{�k�?�  ?E�A4(�C!=q�k�?�(�?Q�A?�
C!��                                    BxW�r  T          @~{�j�H?���?333A$  C =q�j�H?��?B�\A0(�C ��                                    BxW�  "          @����i��?�33=�G�?�G�C�H�i��?�33>.{@�HC�3                                    BxW߾  �          @�G��e�?�{��\)����C���e�?�\)�W
=�AG�Ch�                                    BxW�d  �          @���c33?�33��{��=qC���c33?�zᾊ=q�s�
C�                                    BxW�
  
�          @��
�^�R@Q��(���
=C�\�^�R@�þ�33��(�Cc�                                    BxW�  
�          @�z��W
=@�þ���z�C���W
=@�ýu�c�
C�{                                    BxWV  "          @����Dz�@.{����z�C
xR�Dz�@/\)�����=qC
J=                                    BxW(�  	          @�ff�5@6ff���
�e�C��5@8Q�n{�O�
C��                                    BxW7�  �          @�
=��\@5��  �=qB��q��\@9���	����(�B�W
                                    BxWFH  T          @����7
=@!�?�\)A���C
� �7
=@\)?���A��RC
�                                    BxWT�  �          @����@  @ff�xQ��d��C�3�@  @Q�fff�R�RC�)                                    BxWc�  �          @���J=q?��H��{�ڸRC���J=q?�G���ff���C�H                                    BxWr:  T          @�p���Q�@	���HQ��C33C +���Q�@  �C�
�=B�                                      BxW��  �          @��
�p�@���,���%Q�C�)�p�@{�(��� ��C�{                                    BxW��  �          @�Q��P��@�?0��A�HC33�P��@
=q?@  A-G�Cs3                                    BxW�,  �          @����A�@�?�ffA�z�C��A�@z�?�{A��C��                                    BxW��  �          @��H�P  @�R?���ArffC�=�P  @��?���A�(�C�f                                    BxW�x  T          @����P  @	��?��AtQ�Cz��P  @�?�\)A��RC�{                                    BxW�  �          @�G��Q�@p�?L��A7
=C�q�Q�@(�?\(�AD  C=q                                    BxW��  "          @���P  @�H>8Q�@\)C\)�P  @=q>u@U�Ck�                                    BxW�j  S          @�(��+�@$zῥ�����C(��+�@&ff��p���\)C�{                                    BxW�  T          @�=q�+�@7
=>�=q@\)C)�+�@6ff>�{@��RC.                                    BxW�  �          @����J=q@�H>�
=@�\)C�)�J=q@=q>��@�Q�C��                                    BxW\  T          @����Tz�@
=?O\)A:{Cz��Tz�@ff?Y��AD��C�3                                    BxW"  
�          @�G��R�\@��?�@�C}q�R�\@  ?�A��C��                                    BxW0�  �          @~�R�P  @��?B�\A0Q�C�H�P  @Q�?L��A:ffC��                                    BxW?N  "          @�Q��Mp�@�?uA^=qC���Mp�@
=q?�  Ah(�C�                                    BxWM�  "          @\)�J=q@��?W
=AB=qCk��J=q@  ?aG�AL  C�)                                    BxW\�  T          @�Q��E@�?�G�Aj{C���E@z�?�ffAs�
C.                                    BxWk@  �          @����G
=@
=?s33A[�C�=�G
=@?}p�Ae�C��                                    BxWy�  T          @�  �J=q@��?s33A\  C\)�J=q@  ?}p�Ad��C��                                    BxW��  �          @}p��]p�?�\)?��
Ar�\Cٚ�]p�?�{?��Ax��C�                                    BxW�2  
�          @{��j=q?z�H?}p�Ak33C%��j=q?u?�G�An�RC%@                                     BxW��  �          @u�j=q?Q�?Q�AE��C'c��j=q?O\)?Tz�AHQ�C'��                                    BxW�~  "          @vff�e�?��?^�RAQG�C#)�e�?�=q?aG�AT��C#G�                                    BxW�$  
�          @}p��j=q?��R?O\)A>=qC!L��j=q?�p�?Tz�AA�C!s3                                    BxW��  T          @����hQ�?ٙ��L�ͿB�\C�f�hQ�?ٙ������C��                                    BxW�p  
Z          @|(��i��?��H?O\)A=p�C!�R�i��?���?Q�A@��C!�
                                    BxW�  
�          @z=q�g�?�  ?�ffAx��C$���g�?}p�?��A{\)C$��                                    BxW��  
�          @z�H�b�\?�{?fffAV�\C  �b�\?���?k�AYC�                                    BxWb  
�          @}p��Q�@�\?O\)A=G�C&f�Q�@�?Tz�AA��C=q                                    BxW  �          @|(��Z�H?���?=p�A4��C���Z�H?���?@  A7�C�                                    BxW)�  �          @|(��tzὣ�
?n{AZffC5B��tzὸQ�?n{AZ=qC5\)                                    BxW8T  �          @~{�z=q��G�?&ffA�C5�\�z=q��G�?&ffAC5�H                                    BxWF�  �          @z=q�vff=#�
?&ffA��C3n�vff<�?&ffA��C3z�                                    BxWU�  "          @vff�tz�L��>�Q�@�{C4� �tz�L��>�Q�@�C4Ǯ                                    BxWdF  "          @vff�r�\>�  ?   @�ffC033�r�\>�  ?   @�
=C0=q                                    BxWr�  "          @w
=�l(�?(��?\(�AN�HC)�H�l(�?&ff?\(�AO�C)�                                    BxW��  T          @�G��a�?���?���A�z�C�
�a�?���?���A���C�f                                    BxW�8  �          @�=q�^{?˅?�A�
=C^��^{?˅?�A��Ch�                                    BxW��  �          @���_\)?�  ?�A��C� �_\)?�  ?�A�  CǮ                                    BxW��  �          @�=q�\��?�(�?��
A��RC���\��?�(�?��
A���C��                                    BxW�*  T          @��\�\(�?���?�p�A�{C&f�\(�?���?�p�A�(�C&f                                    BxW��  �          @����XQ�?�
=?�=qA��C�{�XQ�?�
=?�=qA�\)C��                                    BxW�v  T          @����Y��?�G�?�A��C���Y��?�G�?�A�p�C��                                    BxW�  
Z          @����XQ�?�(�?�ffA�
=C�q�XQ�?�p�?�ffA���C��                                    BxW��  
�          @�G��^{?�(�?���A}��C�)�^{?�p�?���A|z�C��                                    BxWh  
�          @����L(�@��?��RA���C5��L(�@��?�p�A�(�C(�                                    BxW  T          @��\�Mp�@Q�?��A�{Cz��Mp�@Q�?�ffA�
=Ch�                                    BxW"�  �          @�z��K�@��?�ffAk�C��K�@=q?��Ah��C޸                                    BxW1Z  �          @����S33@�?�p�A��\C�{�S33@�?�(�A�33C�                                     BxW@   
�          @���P��@�?xQ�AX��C�3�P��@Q�?uAU��C�H                                    BxWN�  T          @���>�R@{?��RA���C^��>�R@�R?�p�A���CB�                                    BxW]L  
�          @�z��G�@ff?��A��C�G�@
=?�=qA��
C��                                    BxWk�  �          @����Dz�@��?�z�A���C���Dz�@=q?��A�G�C��                                    BxWz�  T          @�z��N�R@�?�(�A��C33�N�R@ff?���A�G�C
=                                    BxW�>  �          @�z��Fff@�?��A��C�
�Fff@��?�\)A�z�Cp�                                    BxW��  	�          @����8Q�@.{?�p�A�C�{�8Q�@.�R?���A�z�Cs3                                    BxW��  "          @�z��7
=@%�?��RA���C	���7
=@&ff?�(�A�=qC	�=                                    BxW�0  T          @���:=q@{?\A��
C�f�:=q@\)?��RA�ffCs3                                    BxW��  �          @��\�B�\@{?ǮA�Q�C�H�B�\@\)?��A�
=C��                                    BxW�|  
�          @���A�@  ?�G�A��\C^��A�@G�?�p�A�
=C#�                                    BxW�"  
�          @��\�AG�@$z�?n{AT  C���AG�@%�?fffAK�
Cz�                                    BxW��  �          @���E@=q?�ffAo�
C���E@�?�G�Ag�
C�=                                    BxW�n  
�          @���N{@G�?��A�
=C�{�N{@�\?�{Az{C��                                    BxW  
�          @��Mp�@��?�33A�z�CE�Mp�@�H?�\)AxQ�C�                                    BxW�  
�          @����O\)@  ?�G�A�p�C5��O\)@G�?�(�A�33C�R                                    BxW*`  "          @��
�_\)?�=q?��A�p�CT{�_\)?���?�{A{�C
                                    BxW9  �          @��
�X��?�
=?���A�p�CJ=�X��?��H?��
A�p�C                                      BxWG�  �          @��
�S�
?�\)?�=qA���Cz��S�
?�33?�ffA�z�C�                                   BxWVR              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxWd�  X          @��H�C�
@��?�z�A�  CO\�C�
@�R?�{A|��C�                                    BxWs�  T          @�33�8��@0��?&ffA
=C:��8��@1�?
=A=qC�                                    BxW�D  �          @���333@:�H>#�
@�RC�=�333@:�H=���?�\)C�                                    BxW��  �          @�G��2�\@8��>�33@�p�C�q�2�\@9��>�\)@���C�                                    BxW��  T          @���-p�@>{>���@�p�Cn�-p�@>�R>��@o\)C\)                                    BxW�6  T          @����-p�@;�>�Q�@���C�\�-p�@<(�>�z�@�{C�q                                    BxW��  "          @���2�\@8��>�ff@ʏ\C�R�2�\@9��>\@��
C�H                                    BxW˂  �          @���<��@*�H?J=qA4z�C	�{�<��@,(�?:�HA%C	��                                    BxW�(  
�          @���A�@  ?�
=A���CxR�A�@�?�\)A��C�                                    BxW��  T          @�=q�Mp�@?#�
AC�f�Mp�@
=?z�A  C�R                                    BxW�t  T          @�=q�Vff?У�?���A��RC
=�Vff?�?��
A��C}q                                    BxW  T          @��\�R�\?��H?�ffA�=qC:��R�\@   ?�  A�{C��                                    BxW�  
Z          @��H�N{@�?��Ap��C���N{@�
?�  AbffC\)                                    BxW#f  T          @���N{@p�?�RA�C���N{@{?��@�Q�C��                                    BxW2  
�          @���P��@G�?(��A{C0��P��@�?
=A33C��                                    BxW@�  �          @|���Z=q?�33?�
=A�G�C!n�Z=q?�Q�?�33A���C ��                                    BxWOX  T          @|(��`��?�z�?��A�G�C!���`��?�Q�?�ffA�
=C!:�                                    BxW]�  �          @}p��^�R?Tz�?У�A�ffC&���^�R?aG�?�{A��C%�                                     BxWl�  
�          @����\��?�
=?�  A�
=C!!H�\��?�p�?��HA�ffC Y�                                    BxW{J  �          @����U?�\)?ٙ�A��C�f�U?�
=?�z�A�ffC��                                    BxW��  �          @z�H�Vff?�Q�?�{A��HC c��Vff?��R?���A��C�H                                    BxW��  
�          @w��R�\?��\?\A�(�C���R�\?���?�(�A��RC+�                                    BxW�<  T          @z=q�[�?\(�?�33A�=qC%�)�[�?k�?�\)A�ffC%�                                    BxW��  "          @z�H�X��?c�
?�(�AиRC%E�X��?s33?�Q�Ạ�C$^�                                    BxWĈ  T          @z=q�X��?�Q�?��\A���C��X��?�p�?�(�A�p�Cn                                    BxW�.  T          @z=q�`��?�z�?z�HAhz�C&f�`��?�Q�?n{A\z�C��                                    BxW��  T          @z�H�\(�?��?�ffAx��C��\(�?���?}p�Ak\)Cp�                                    BxW�z  �          @����Dz�@��?^�RAH(�Ch��Dz�@�R?G�A3�C�                                    BxW�   �          @����Tz�?�G�?���A�  C� �Tz�?���?�G�A���C�3                                    BxW�  �          @\)�Z�H?�(�?�A��CǮ�Z�H?\?�\)A��HC�                                    BxWl  �          @�G��U�?�p�?��HA��C�H�U�?��
?��A��C��                                    BxW+  
�          @�G��P��?�?�A�z�C� �P��?�(�?��A��C��                                    BxW9�  �          @�  �R�\?�{?��A�33C���R�\?�33?��RA�ffC�f                                    BxWH^  
�          @\)�QG�?�?���A�(�C�3�QG�?��?��
A�G�C                                    BxWW  �          @}p��U?�?aG�AL��C\�U?��H?L��A:=qC�H                                    BxWe�  �          @y���S�
?���?s33A`��C+��S�
?�{?^�RAN�HC��                                    BxWtP  
Z          @u��X��?z�H?��RA���C#�=�X��?��?���A�(�C"ٚ                                    BxW��  
�          @h���N{?W
=?�p�A��C%J=�N{?fff?�Q�A��\C$G�                                    BxW��  �          @hQ��C33?�  ?ٙ�A��
C!޸�C33?���?�z�AٮC �                                    BxW�B  !          @j=q�>�R?�{?�ffA�ffC���>�R?�
=?�  A�\)CY�                                    BxW��  �          @g
=�>�R?�Q�?�z�AۮC:��>�R?�G�?�{A�(�C�                                    BxW��  �          @k��@  ?���?�G�A��C&f�@  ?��
?��HA���C�                                    BxW�4  "          @fff�7�?��?���A��RC\)�7�?�(�?�\A�
=C�                                    BxW��  �          @g��:=q?�z�?�A�C:��:=q?��R?�G�A�  C��                                    BxW�            @\���*=q?s33?�p�B33C Q��*=q?��?�Q�B��C��                                    BxW�&  "          @Y���p�?L��@�B ��C!�)�p�?fff@��B33C��                                    BxW�  �          @\(��(Q�?@  @�B
=C$)�(Q�?W
=?��RB��C"B�                                    BxWr  
�          @Fff���
=�G�@"�\B[�
C0�����
>k�@"�\BZ�
C,�q                                    BxW$  �          @QG���\=���@.�RBd(�C0�ÿ�\>k�@.�RBc�C,�\                                    BxW2�  �          @W��   ��\)@,��BU��C6��   =u@,��BU�
C2ff                                    BxWAd  "          @c�
���>�\)@'
=B<�C-p����>���@%B;(�C*�                                     BxWP
  �          @b�\�!�?�@��B*��C(
�!�?&ff@
=B(p�C%��                                    BxW^�  
�          @a��"�\?z�@ffB(=qC')�"�\?333@z�B%z�C$��                                    BxWmV  T          @e��p�?n{@�B*�
CQ��p�?�ff@Q�B&�\C�                                    BxW{�  
�          @g
=� ��?G�@{B,z�C"� ��?fff@�HB(�
C =q                                    BxW��  "          @e�1G�?��?�=qA��HCL��1G�?�(�?��RAĸRC�                                    BxW�H  
Z          @`  �%�?���?�=qA��RC���%�@ ��?�p�A�ffC�                                    BxW��  
�          @c�
�{@�?\AɅC��{@	��?�z�A�{C
��                                    BxW��  	I          @j=q�!�?�
=?�A��HC���!�@G�?��HA�(�CaH                                   BxW�:              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxW��              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxW�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxW�,              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxW��              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxWx              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxW              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxW+�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxW:j              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxWI              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxWW�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxWf\              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxWu   2          @dz��(Q�@�?�A���C#��(Q�@�?�ffA���CJ=                                    BxW��  
�          @`  �(Q�@33?��A��C)�(Q�@
=?k�At��CT{                                    BxW�N  "          @dz��0��@G�?uAzffC�=�0��@z�?W
=AZ{C�                                    BxW��  �          @hQ��7�?��R?xQ�Aw�CL��7�@�\?Y��AXQ�C��                                    BxW��  
Z          @qG��G
=?�=q?�ffA�{C���G
=?��?p��Ah  C�q                                    BxW�@  �          @j�H�B�\?�\)?O\)AL��Cs3�B�\?�?333A/�C�{                                    BxW��  �          @l���B�\@   ?!G�Ap�C���B�\@�?�\@�p�CG�                                    BxWی  "          @p  �N{?�=q?�\@���Cp��N{?���>Ǯ@���C�                                    BxW�2  �          @`  � ��@�>\@љ�C� ��@��>�  @�Q�C��                                    BxW��  �          @HQ����@*=q�#�
�8��B�{����@)�����
��{B�\)                                    BxW~  
Z          @=p����@*�H��
=��\B�=q���@(�ÿz��7\)B�q                                    BxW$  "          @%��c�
@z�0���w33B�  �c�
@녿Tz����\B���                                    BxW$�  �          @&ff�Y��@�����D  B�uÿY��@��333�z�RB�\                                    BxW3p  	`          @�ÿz�@�׾��H�<z�B��z�@{�!G��u�B�aH                                    BxWB  "          @�R���@z����d��B�uÿ��@녿.{���\B�\                                    BxWP�  
�          @ff�+�@
=q�
=q�U��B�\)�+�@Q�.{��
=B���                                    BxW_b  T          @(����?��s33��(�Bî���?��Ϳ������B�=q                                    BxWn  T          @{>��R?�(���ff��RB�\)>��R?У׿�z��!p�B�B�                                    BxW|�  �          @>��R?�������z�B�>��R?�Q�Ǯ�)G�B��H                                    BxW�T  �          @�>�p�@ �׿����ffB�{>�p�?�Q쿜(���(�B�G�                                    BxW��  T          @��>aG�?�
=������B�8R>aG�?�=q��  ���B��\                                    BxW��  �          @��>��?�Q�k���  B�=q>��?�\)�����\B��                                    BxW�F  T          @  ��?�Q쿌����Q�B�{��?�{��p���B�aH                                    BxW��  "          @���\)?��ͿaG�����B�W
��\)?����  ���B��                                     BxWԒ  "          @�L��?��s33��=qB�=q�L��?�\������p�B�\)                                    BxW�8  
Z          @8Q�>�?��R���H�/33B��>�?�����>p�B�
=                                    BxW��  �          @E�?0��?�(���R�;�RB�aH?0��?��
=�J�B�G�                                    BxW �  �          @P��?
=@�\�p��D��B��=?
=?�{�%�S�B��                                    BxW*  
�          @AG�>�  @�
���9�RB�L�>�  ?�33�z��I�B�.                                    BxW�  �          @L��?�@
=��<�B�?�?�Q���R�K��B��3                                    BxW,v  	�          @Mp�?�@(��G��4��B���?�@G���H�D\)B��3                                    BxW;  �          @^�R?(��@  � ���;B�aH?(��@�
�*=q�K33B��                                    BxWI�  �          @qG�?��@)���Q���RB�p�?��@{�#�
�-�B���                                    BxWXh  
�          @u�?�?�=q�B�\�T{B-��?�?�{�I���^�
B�                                    BxWg  �          @{�?��?�{�J�H�XQ�B1�?��?����Q��cQ�B 
=                                    BxWu�  "          @��
?У�?����S33�V{B@�H?У�?�=q�[��b  B0��                                    BxW�Z  T          @�  ?�=q?�z��Q��\33B9��?�=q?��X���gB'�                                    BxW�   
�          @U?���    �=p��z�C���?����aG��=p��~=qC���                                    BxW��  
�          @\��?��=L���<(��sp�?��
?�녾#�
�;��r�
C�(�                                    BxW�L  T          @\)@  ?�=q�G
=�L��A�\)@  ?Y���K��S��A��
                                    BxW��  T          @u?�33?�G��B�\�TG�B�?�33?���G��]
=A�                                    BxW͘  T          @z=q?޸R?�{�C33�PB+=q?޸R?����J=q�[B�H                                    BxW�>  
�          @z�H@	��?���>{�F�RA���@	��?���C�
�O  A�z�                                    BxW��  T          @{�@�?����G
=�RA㙚@�?fff�L(��Z33A�                                      BxW��  
Z          @\)@(�?��
�J�H�RffA�  @(�?J=q�O\)�Y  A�\)                                    BxW0  �          @~{@z�?k��E�LQ�A�
=@z�?0���I���Q��A�                                    BxW�  �          @\)@p�?L���A��E�HA�
=@p�?��E��Jz�AN�R                                    BxW%|  �          @�=q@,(�?�  �8Q��4ffA���@,(�?G��<���9��A���                                    BxW4"  
�          @�ff@:=q?:�H�;��1  A_�
@:=q?   �>{�4�\Az�                                    BxWB�  
�          @�=q@!녽�\)�J�H�M��C�&f@!녾��
�I���L=qC�p�                                    BxWQn  T          @z=q@녾\�HQ��Vz�C�Ff@녿�R�Fff�R�RC�k�                                    BxW`  
�          @q�@{��\)�A��V33C�g�@{��\�@  �S�C���                                    BxWn�  �          @n{@ff>���4z��G�H@��@ff=#�
�5��I
=?��                                    BxW}`  
�          @j�H@�R?��)���9(�A?�@�R>��
�+��<(�@�33                                    BxW�  T          @x��@/\)>��,���0z�A�R@/\)>�=q�.�R�2�H@��                                    BxW��  
�          @}p�@1녽��4z��5��C��
@1녾�{�333�4�C�z�                                    BxW�R  
�          @z=q@1녿8Q��(Q��)��C���@1녿n{�#�
�$�C�Ǯ                                    BxW��  �          @n{@/\)�
=��H�#G�C���@/\)�G�����C�{                                    BxWƞ  T          @U?��\?��	���0��B[�?��\?У����?p�BP�
                                    BxW�D  T          @Z=q@��>�z��ff�1G�@��
@��=����
=�2@(�                                    BxW��  T          @u�@���J�H�bG�C��=@��\)�I���`�C�#�                                    BxW�  "          @�G�@�����N{�U�C��)@����z��Mp��TG�C��3                                    BxW6  
�          @��
@.�R=��E��A@"�\@.�R���E��A�RC��3                                    BxW�  �          @�@/\)<#�
�I���D  >��R@/\)�u�H���CG�C���                                    BxW�  "          @���@ff��z��c33�lQ�C�\@ff���`���h��C�`                                     BxW-(  "          @��@!G��
=�I���JffC�p�@!G��W
=�Fff�EffC��                                    BxW;�  
�          @u�@\)�B�\�333�<{C���@\)�z�H�.�R�6  C�=q                                    BxWJt  "          @l��@��u�*=q�8p�C��@����$z��0�
C��\                                    BxWY  !          @l(�@(���Q��!��,=qC�H@(�������#�\C�(�                                    BxWg�  
�          @w
=@'�����\)� �
C�}q@'����
����C��{                                    BxWvf  �          @vff@-p������
��C�e@-p���=q�(���RC���                                    BxW�  
(          @��@(�ÿ���8Q��5�HC�4{@(�ÿ��\�2�\�.\)C��                                    BxW��  T          @xQ�@4zῌ���=q��C�Y�@4zῥ���
���C���                                    BxW�X  �          @}p�@,(��z��5��7=qC��@,(��O\)�1G��2�C��)                                    BxW��  "          @\)@6ff��G��0  �.C���@6ff�(���-p��+33C�u�                                    BxW��  �          @x��@4z���&ff�(C�Ф@4z�:�H�#33�$�C���                                    BxW�J  T          @hQ�@�
�.{�*=q�?p�C���@�
�c�
�&ff�9\)C�l�                                    BxW��  
�          @\��@G��Tz����4�C�@G����\�ff�-\)C��H                                    BxW�  �          @\(�?��þ�=q�8Q��e�C��H?��ÿ�\�6ff�a��C�.                                    BxW�<  �          @p  ?޸R��ff�I���q33C���?޸R�5�Fff�k�C��3                                    BxW�  �          @q�@���33�,(��6(�C��@���\)�%��-33C���                                    BxW�  �          @���@-p������,���'z�C���@-p�����%���C�'�                                    BxW&.  �          @�z�@3�
�z�H�5�.C�c�@3�
���H�0  �'��C�Q�                                    BxW4�  �          @�{@�R����E��=(�C�Y�@�R����<���3(�C�9�                                    BxWCz  T          @��R@'
=��(��<���2p�C�T{@'
=���H�4z��(p�C�b�                                    BxWR   T          @�{@&ff��z��3�
�)G�C��3@&ff����*=q�ffC��R                                    BxW`�  �          @��@-p������-p����C��@-p��33�"�\�ffC�u�                                    BxWol  T          @�  @5���(��{�(�C��R@5��
�H��\�C�E                                    BxW~  
�          @��@4z����� �����C�8R@4z���
=���C���                                    BxW��  "          @l��@   ����Q�� ffC�!H@   ��G������
C�l�                                    BxW�^  �          @n�R@����H�
=�  C��H@���33����{C�c�                                    BxW�  �          @l��@p���\)�\)�(��C��@p�������  C�G�                                    BxW��  "          @���@0  ��(��ff��HC���@0  ������{C��=                                    BxW�P  �          @w
=@C33��(���=q����C�J=@C33���Ϳ�
=��33C�b�                                    BxW��  �          @l(�@N{��������33C�!H@N{��G��z�H�u�C�y�                                    BxW�  "          @l��@U����@  �<z�C�w
@U���p��!G��G�C��                                    BxW�B  T          @p  @J�H��=q�J=q�B{C��@J�H��녿!G��ffC��                                     BxW�  T          @mp�@8��� �׿����{C���@8����k��f�RC��                                    BxW�  
�          @dz�@+�����\)���\C��@+��
�H�p���t(�C�xR                                    BxW4  �          @a�@-p��녿xQ��~�RC���@-p��ff�J=q�N�HC�q                                    BxW-�  
�          @e�@&ff�p������C�޸@&ff��\�\(��`  C�Y�                                    BxW<�  T          @h��@(�����n{�l��C�>�@(���=q�:�H�8  C�Ф                                    BxWK&  T          @l(�@�R��H��G���Q�C��@�R�!G���ff��ffC�N                                    BxWY�  �          @qG�@   ��Ϳ�����HC��3@   �#�
�����HC�0�                                    BxWhr  �          @p  @��)����
=���\C��f@��/\)�p���ip�C�j=                                    BxWw  �          @dz�@��(�ÿ}p���33C��=@��.{�B�\�D��C�e                                    BxW��  �          @dz�?�
=�5��u�{�C�(�?�
=�9���8Q��9C���                                    BxW�d  �          @`  @G��'��������C���@G��,�Ϳ^�R�h  C�c�                                    BxW�
  �          @X��@����Ϳp����=qC���@���!녿8Q��D  C�                                      BxW��  T          @Y��?���"�\��z���p�C�G�?���(Q�p����33C���                                    BxW�V  
�          @Y��@���׿:�H�G�C���@���
�����C�0�                                    BxW��  
�          @W
=@����R�.{�;�
C��R@���녾���C�Ff                                    BxWݢ  �          @U�@G�� �׿.{�?�C�n@G��#�
����C�'�                                    BxW�H  �          @XQ�?�=q�'
=�����Q�C��f?�=q�,�ͿY���j�HC��                                    BxW��  
�          @U�?���!G��������C�w
?���&ff�^�R�q�C�H                                    BxW 	�  T          @U?��&ff��G���Q�C��q?��+��G��V�HC�7
                                    BxW :  
�          @G�?��
�Q�xQ����C�]q?��
�p��B�\�a��C��\                                    BxW &�  T          ?���?�G����ÿ���p�C��f?�G���{����V�HC�Z�                                    BxW 5�  �          ?0�׽u>8Q�>���Bc�RB��H�u>W
=>�\)BP�\BԨ�                                    BxW D,  T          @�
����?�p�?E�A�{C쿙��?��?#�
A���B��)                                    BxW R�  
�          @-p���  @33>��@���Cs3��  @z�=��
?ٙ�C=q                                    BxW ax  
�          @,�Ϳ�z�@33?
=qA:ffC��z�@>�Q�@�\)C �=                                    BxW p  �          @-p���G�?��R?\)A=�Cuÿ�G�@�>\A ��C�H                                    BxW ~�  T          @'
=��G�?�=q>��A&�HC�f��G�?�{>��R@�G�CaH                                    BxW �j  "          @'
=���H?�>���AG�C����H?���>k�@��CE                                    BxW �  �          @333��Q�@\)>B�\@z�HB�{��Q�@\)�#�
����B��H                                    BxW ��  �          @:�H���
@G�>�G�A(�C ����
@33>u@���B�z�                                    BxW �\  �          @C33�Ǯ@'
=>u@�=qB���Ǯ@'�<#�
>uB�p�                                    BxW �  �          @@  ��z�@(Q�=���?�(�B�zῴz�@(Q���{B�                                     BxW ֨  T          @B�\��\)@4z�>\)@(Q�B�uÿ�\)@4z��G��33B�p�                                    BxW �N  
�          @<�Ϳ�\)@-p�>B�\@n{B��
��\)@.{�L�Ϳn{B�q                                    BxW ��  T          @=p�����@0��=�Q�?�(�B�k�����@0�׾���>�RB�u�                                    BxW!�  �          @<(��333@5�u��\)Bϙ��333@4zᾞ�R��Q�B�                                    BxW!@  �          @B�\�.{@8��?
=A4Q�BΞ��.{@;�>�{@˅B�B�                                    BxW!�  �          @Dz�5@5?c�
A��B��f�5@:=q?#�
AA�B�G�                                    BxW!.�  T          @@  �8Q�@1�?W
=A��HB���8Q�@5?��A7�B�z�                                    BxW!=2  "          @<�Ϳz�@1G�?Q�A�{Bˏ\�z�@5�?z�A5�B�\                                    BxW!K�  �          @3�
���@*�H?(�AG\)B�(����@.{>�p�@�G�B�Ǯ                                    BxW!Z~  �          @\)�^�R@�
>�=q@ƸRB�Q�^�R@�=u?��RB�{                                    BxW!i$  
�          @!G�����@  >aG�@��\B������@��<��
?��B��                                    BxW!w�  
�          @Q��?˅=u?��
C^���?˅���
����CaH                                    BxW!�p  �          @&ff��
=?�
=>��A\)C
���
=?��H>�=q@�z�C
z�                                    BxW!�  "          @%��(�?�ff?&ffAhQ�C޸��(�?���?�\A6{C��                                    BxW!��  
�          @   ��p�?�(�?�\A<��C@ ��p�?�G�>�Q�A�C�H                                    BxW!�b  >          @�R��Q�?�������C{��Q�?}p��(�����C��                                    BxW!�  
�          @\)����>aG�����EQ�C,.����=L�Ϳ���G
=C2\                                    BxW!Ϯ  T          @   ����>��H�����OffC!5ÿ���>��
���R�T�
C'��                                    BxW!�T  
�          @%����>�ff�z��W
=C"�ÿ���>�=q�ff�[�HC)�                                    BxW!��  �          @{�k����H�Q��33CP��k��.{�z��tp�CXE                                    BxW!��  
�          @%�@  �����z���CP@ �@  �(���{CZٚ                                    BxW"
F  
Z          @&ff�fff�#�
�=q{C4޸�fff�aG����ǮCB�                                    BxW"�  
�          @'��p��>8Q����z�C)J=�p�׼��=q=qC5��                                    BxW"'�  
�          @(�ÿ�=q>�Q��33�33C!�)��=q>#�
�z��C+�=                                    BxW"68  
�          @.{����>�  �(��X{C*޸����=u�p��Z
=C1��                                    BxW"D�  �          @5���?s33����m=qC
!H���?=p���R�y�C.                                    BxW"S�  �          @6ff����?�\)��R�P33B��Ϳ���?���_�CQ�                                    BxW"b*  "          @5����R?���
=q�H\)C=q���R?�\)����V\)C	Ǯ                                    BxW"p�  
�          @-p��n{?�  �33�a=qC#׿n{?Q��Q��o�C
�f                                    BxW"v  "          @:=q��\����0����Cc��\�J=q�,���qCl�R                                    BxW"�  "          @3�
���>aG��&ffW
C(쿅��#�
�&ff\)C4O\                                    BxW"��  "          @;����>�33�,(�ǮC!�{���=�G��-p��fC-�f                                    BxW"�h  
�          @@  ��  >�=q�,��p�C'���  =#�
�-p���C2aH                                    BxW"�  "          @A녿��H>���*�H.C���H>u�,����C(�f                                    BxW"ȴ  �          @B�\��\)?�G��\)�^��C�
��\)?J=q�$z��iz�C��                                    BxW"�Z  "          @I����  >�33�/\)�rQ�C&�H��  =�G��0���u\)C/��                                    BxW"�   �          @L(����׽#�
�6ffG�C5ٚ���׾�z��5�~(�C?޸                                    BxW"��  
�          @J=q���R��  �0���uQ�C=�׿��R���H�.�R�p�\CF5�                                    BxW#L  
�          @;�>����
�ٙ��p�C�3>����Ϳ��R��  C�˅                                    BxW#�  �          @<�ͼ��
�Q��z��'�C������
��\��(���
C���                                    BxW# �  �          @E��#�
�z��   �/�\C��q�#�
��R�����C��                                    BxW#/>  "          @N�R?5��  �/\)�k{C���?5��p��&ff�Y�C�4{                                    BxW#=�  "          @J�H?s33��p��/\)�q�RC��q?s33���H�'��a�RC���                                    BxW#L�  �          @J�H?B�\���
�8��B�C�8R?B�\���\�2�\�x(�C�l�                                    BxW#[0  "          @J�H>aG����\�>�Rk�C��>aG����\�8Q���C��                                    BxW#i�  �          @I���L�Ϳh���@��=qC�xR�L�Ϳ�z��:�Hz�C��                                    BxW#x|  "          @J=q>#�
�fff�?\)  C��H>#�
��33�:=q\)C��{                                    BxW#�"  "          @Z=q?+��\(��Mp��\C�޸?+������G�z�C�B�                                    BxW#��  T          @XQ�?(�ÿY���K���C���?(�ÿ�\)�Fff�=C�>�                                    BxW#�n  
�          @P��?0�׿���:=q�C��3?0�׿����333�t\)C�AH                                    BxW#�  �          @G
=?n{��  �*�H�nC�\)?n{��(��#33�^��C�/\                                    BxW#��  T          @XQ�?��Ϳ��R�-p��^z�C�(�?��Ϳ�(��%��N�C�L�                                    BxW#�`  �          @`  ?�\��
�(���C�W
?�\�\)� �����C�.                                    BxW#�  "          @c33?��   �
=q�=qC�Q�?��
�H���R�C�!H                                    BxW#��  T          @b�\@�Ϳ�ff����RC��{@�Ϳ�p��   �33C�l�                                    BxW#�R  
�          @b�\?ٙ�����
�H�%�\C��f?ٙ����H� ���\)C�p�                                    BxW$
�  �          @j�H?}p���p��L(��s=qC��?}p��޸R�C�
�b�C��                                    BxW$�  �          @l��?333���
�XQ�W
C�o\?333��ff�P���y�\C�/\                                    BxW$(D  �          @j�H?�z῕�J=q�o�RC�8R?�zῷ
=�C�
�b��C�]q                                    BxW$6�  T          @j�H@   �����7��N{C���@   ��=q�1G��D33C�.                                    BxW$E�  
Z          @j�H@G������0  �A��C��R@G������(Q��6�C��=                                    BxW$T6  �          @mp�?��H����N{�rC��
?��H��=q�G��fC��H                                    BxW$b�  
�          @k�?��ͿO\)�Tz�aHC��=?��Ϳ�=q�P  �w��C��=                                    BxW$q�  
�          @l(�?��
��G��L(��p�RC�H�?��
���\�Fff�ez�C�'�                                    BxW$�(  "          @n{@z�=p��>{�V=qC�+�@z�z�H�9���O  C�P�                                    BxW$��  
�          @n�R?��H��
=�5�FQ�C��{?��H��33�.{�:�C��                                    BxW$�t  	�          @o\)?�\��p��;��O{C��?�\��(��333�B�C��                                    BxW$�  
�          @q�?�������A��U�\C��?����
�7��F{C�<)                                    BxW$��  �          @p  ?�Q��ff�L(��jG�C��R?�Q��ff�C�
�[
=C��=                                    BxW$�f  R          @k�?333��ff�N�R�y33C�)?333��ff�E�h  C���                                    BxW$�   �          @k�?Q����(���7{C�B�?Q��(�����%p�C���                                    BxW$�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxW$�X              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxW%�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxW%�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxW%!J              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxW%/�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxW%>�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxW%M<              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxW%[�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxW%j�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxW%y.              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxW%��              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxW%�z              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxW%�               ��O���O���O���O���O���O���O���O���O���O���O�                                  BxW%��              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxW%�l              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxW%�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxW%߸              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxW%�^              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxW%�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxW&�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxW&P              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxW&(�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxW&7�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxW&FB              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxW&T�  �          @tz�?Q��
�H����7��C�U�?Q��ff��R�'�\C��H                                    BxW&c�  "          @tz�?L��?�
=�`���3B^?L��?k��e�p�BB                                    BxW&r4  "          @z=q?0��?h���n�R�HBS��?0��?#�
�r�\.B*ff                                    BxW&��  T          @w�?.{�0���J�H
=C�]q?.{�k��G
=��C�E                                    BxW&��  �          @hQ�?Y���   �\)�-=qC�XR?Y���+��33�=qC��                                    BxW&�&  �          @l(�?�ff�  �.{�>�\C�z�?�ff�(��#33�/=qC���                                    BxW&��  �          @n�R?������H�:�H�OffC�� ?����
=q�1G��A  C�~�                                    BxW&�r  �          @l��?�����33�>{�V�C��f?�����{�5�IQ�C���                                    BxW&�  "          @mp�?�
=���>�R�VC�H�?�
=����7
=�I�HC��q                                    BxW&ؾ  
�          @n�R?�=q���
�E�b��C���?�=q��  �>{�U�HC��=                                    BxW&�d  T          @~{?�p���=q�Q��`Q�C��?�p���
�H���Rz�C�~�                                    BxW&�
  �          @~{?��R�8Q��e�k�C�(�?��R�xQ��aG��|��C�y�                                    BxW'�  T          @|��?�ff��G��Y���np�C�g�?�ff�޸R�Q��aC�]q                                    BxW'V  �          @���?У׿��\�`���u�C��?У׿�G��[��k�HC�+�                                    BxW'!�  T          @�Q�@p��J=q�O\)�XQ�C�9�@p���G��K��RG�C��R                                    BxW'0�  
Z          @��\?�Q�W
=�hQ�C�~�?�Q���fff8RC�^�                                    BxW'?H  �          @��?�p�>k��z=q�{A+�?�p��#�
�z=q33C��                                    BxW'M�  
�          @���?aG�>��x��#�A�ff?aG�>W
=�z�H  AT��                                    BxW'\�  
�          @�  ?���?p���o\)G�B%�?���?.{�s33
=B�                                    BxW'k:  T          @u�?�z�?�p��U��q��BO�?�z�?�  �Z�H�}��B<                                    BxW'y�  �          @\(�?   ?�  �AG��y\)B��?   ?���G
=
=B���                                    BxW'��  T          @h��>u@Tz�\��{B�>u@Mp��޸R���B�z�                                    BxW'�,  T          @k�>�\)@C33�33�=qB�k�>�\)@9���  ���B��f                                    BxW'��  T          @p��?   @B�\�	���Q�B�ff?   @8���ff��B�u�                                    BxW'�x  �          @p  ?\)@@���
�H�z�B��)?\)@7
=����B�Ǯ                                    BxW'�  "          @i��?��@7��
�H�p�B�#�?��@.{�
=�!ffB���                                    BxW'��  
�          @\��?O\)@   �Q��p�B�=q?O\)@ff��\�*�
B�(�                                    BxW'�j  �          @[�?��
@\)���/�B�?��
@�� ���=Q�B~�H                                    BxW'�  
�          @Tz�?�p�?����!G��E�B;  ?�p�?�33�'��P�\B.
=                                    BxW'��  �          @P��?��\?��
=�9�B]��?��\?�Q��{�F  BS�                                    BxW(\  �          @E?�z�?�p��33�K��B$�\?�z�?�=q�Q��U
=B�
                                    BxW(  
�          @<(�?˅?.{�
=�VA�p�?˅?�����\  A�
=                                    BxW()�  "          @(��?�ff?�\���d��A���?�ff>�p��p��i��A|��                                    BxW(8N  
�          @(��?��þǮ�
=�c�C���?��ÿ����^\)C�#�                                    BxW(F�  "          @AG�?���?�
=����<�
B2�R?���?����\�FB&��                                    BxW(U�  "          @4z�?��?����ff�Ap�B9�?��?��H���K�\B-{                                    BxW(d@  
�          @Q�?�  ?��ÿ��H�8B"ff?�  ?u���
�A��B\)                                    BxW(r�  
�          @  ?��?k���=q�3=qB�H?��?Q녿У��;33B�R                                    BxW(��  �          @(�?�33?n{�޸R�7�HBQ�?�33?Q녿�ff�?\)A�
=                                    BxW(�2  �          @
=?�  ?�ff��Q��7��B �
?�  ?s33��  �@��B{                                    BxW(��  �          @�\?�
=?��ÿ���/��B(?�
=?xQ�����9{B(�                                    BxW(�~  �          @ ��?�{?L�Ϳ�Q��:G�B  ?�{?5��p��B{Bp�                                    BxW(�$  �          ?�
=?��?�������	p�B,ff?��?�  ������RB%�                                    BxW(��            @*�H?�(�@���\)��\)Bn?�(�@ �׿��R���HBkG�                                    BxW(�p  �          @O\)?(��@>{�L���j�\B��?(��@:�H�xQ���B��                                    BxW(�  �          @dz�?�z�@.�R�����љ�Bj��?�z�@(Q��p���\Bf�                                    BxW(��  �          @�33@'
=@	���33��B  @'
=@ ����H�
=B��                                    BxW)b  �          @�\)@   @3�
��\)��BA�R@   @,������B=                                      BxW)  �          @�\)@�@Fff��  ����BO@�@@�׿�
=���BLp�                                    BxW)"�  
�          @�  @�\@B�\�����(�BS��@�\@:�H�33����BO��                                    BxW)1T  �          @�=q@(�@J=q�����=qBQ  @(�@C�
���ƸRBMz�                                    BxW)?�  T          @�(�@p�@R�\��  ��Q�BT�@p�@L�Ϳ�
=���HBQ��                                    BxW)N�  �          @��
@*=q@Q녿�{�k�
BK��@*=q@Mp������BIz�                                    BxW)]F  �          @��@%�@Tz�xQ��O�
BP�@%�@P�׿�33�w�
BN�                                    BxW)k�  �          @�z�@�@I����=q��\)BX�@�@E���G���BU�                                    BxW)z�  �          @�{@	��@X�ÿ�z���33Bfz�@	��@TzΎ���Q�Bd\)                                    BxW)�8  �          @�33@*�H@W
=�����BM��@*�H@Tz�5�33BL�
                                    BxW)��  T          @���@��@g
=�333�(�BmQ�@��@dz�fff�B�\Bl�                                    BxW)��  �          @���@�@p�׾�(����RBv�@�@n�R�!G��ffBu��                                    BxW)�*  T          @�33@�@s�
�W
=�3�
Bu��@�@r�\�����
=Bu33                                    BxW)��  �          @��@(�@`��>�?ٙ�B\z�@(�@`�׽u�E�B\�                                    BxW)�v  �          @�
=@  @c�
�L�Ϳ�RBf@  @c33�k��H��Bf�                                    BxW)�  �          @�@�@e��=q�p  Bn  @�@dz���ə�Bmz�                                    BxW)��  �          @�{@
=q@dz�#�
�8Q�Bk�@
=q@dz�L���,(�Bkff                                    BxW)�h  �          @�ff@G�@aG�>��@_\)Bd�R@G�@a�=��
?�ffBd��                                    BxW*  �          @�ff@33@\��?#�
A  Ba  @33@^{>�@�33Ba�
                                    BxW*�  T          @���@��@^{?B�\A)�Bi{@��@`  ?
=A=qBj{                                    BxW**Z  �          @�z�?��@h��?J=qA0��B{?��@j�H?(�AQ�B��                                    BxW*9   T          @��\?���@Y��?�
=A���B��q?���@^�R?�G�A�  B���                                    BxW*G�  �          @�ff?�33@n�R?��A�G�B�?�33@r�\?���Ax��B���                                    BxW*VL  �          @���?�=q@[�?��HA�p�B���?�=q@`  ?�ffA�\)B�p�                                    BxW*d�  �          @�  ?��@S33?�ffA��RB~=q?��@W�?�33A�33B�\                                    BxW*s�  �          @�33@   @C33?�A�Bc
=@   @HQ�?�Q�AîBe�
                                    BxW*�>  T          @�Q�@
=@6ff?���Aܣ�BU�R@
=@;�?�(�A�p�BX�
                                    BxW*��  
�          @~{?���@G
=?\A�z�Bg�?���@J�H?���A��\Bi�H                                    BxW*��  T          @�Q�@ ��@P��?�33A��BiG�@ ��@S�
?�  Ag�Bj��                                    BxW*�0  �          @�33?�@[�?�\)A~{Bs
=?�@^�R?uAY�Bt\)                                    BxW*��  �          @��\?�{@]p�?�=qAv�\Bv��?�{@`  ?n{ARffBx                                      BxW*�|  �          @��?��@Z=q?�z�A��\Bt�?��@]p�?�G�Ae��Bv                                      BxW*�"  �          @���?�@\(�?uA\z�Bw��?�@^�R?O\)A9�Bx��                                    BxW*��  �          @�Q�?˅@^{?�Q�A���B���?˅@aG�?��AqG�B�=q                                    BxW*�n  �          @���?�ff@^{?h��AR{BzQ�?�ff@`��?B�\A/\)B{G�                                    BxW+  �          @l��?Tz�@C�
?���A�Q�B��?Tz�@HQ�?�(�A߮B�aH                                    BxW+�  �          @l(�?aG�@E�?�ffA�RB�#�?aG�@I��?�A�Q�B���                                    BxW+#`  �          @p��?�@H��?�p�A�  B�G�?�@L��?���A��RB�\                                    BxW+2  �          @p  ?�
=@E�?���A�p�B��?�
=@H��?�(�A�z�B�                                      BxW+@�  �          @p��?�@K�?���Aʏ\B���?�@P  ?�p�A�G�B�W
                                    BxW+OR  �          @vff?�\)@QG�?�(�A��\B�k�?�\)@Tz�?��A�B��                                    BxW+]�  �          @~{?���@S�
?��RA���B�u�?���@W
=?�{A��RB�33                                    BxW+l�  �          @}p�?�@Mp�?�
=A��
Brp�?�@P��?��A�ffBt
=                                    BxW+{D  �          @~{?��@W�?��A�{B��?��@Z=q?�
=A�Q�B��R                                    BxW+��  T          @|��?�33@fff?O\)A=�B�aH?�33@hQ�?+�A��B��                                    BxW+��  �          @vff?��@dz�?��A(�B�B�?��@e>��@�Q�B�u�                                    BxW+�6  �          @s�
?�z�@P��?G�A@(�B|�?�z�@R�\?(��A"=qB|�H                                    BxW+��  �          @dz�?�G�@S�
>#�
@*=qB�� ?�G�@Tz�=#�
?8Q�B��=                                    BxW+Ă  �          @c33?�ff@XQ�>��@�ffB���?�ff@X��>\)@  B��                                    BxW+�(              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxW+��              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxW+�t  6          @hQ�?�\)@P  =��
?��B~=q?�\)@P  ����B~G�                                    BxW+�  �          @c33?�=q@P��>�=q@��\B�� ?�=q@P��>#�
@#33B��{                                    BxW,�  �          @`  ?xQ�@Vff>�=q@�33B���?xQ�@W
=>��@!�B��                                    BxW,f  �          @J�H?=p�@'��s33��ffB���?=p�@%������RB�=q                                    BxW,+  �          @\��?��\@J=q>�=q@���B�Q�?��\@J�H>.{@6ffB�k�                                    BxW,9�  �          @c�
?�p�@N{>L��@S33B�Ǯ?�p�@N�R=���?�B��)                                    BxW,HX  �          @g
=?��R@P  =�?���B���?��R@P��<��
>�{B��3                                    BxW,V�  �          @fff?�@C�
>\@�z�Bk��?�@Dz�>�z�@��Bl                                      BxW,e�  �          @c33?���@I��>aG�@c33B|
=?���@I��>�@�\B|33                                    BxW,tJ  �          @`��?��\@P  >�  @��
B�aH?��\@P��>#�
@#�
B�p�                                    BxW,��  �          @`  ?���@I��>�p�@��HB�p�?���@J=q>�\)@��B��\                                    BxW,��  �          @j=q@   @A�>�Q�@�33Bb33@   @B�\>�=q@�  Bb�                                    BxW,�<  �          @n{@Q�@@��>�z�@�\)B[=q@Q�@AG�>W
=@L(�B[z�                                    BxW,��  T          @r�\@�@Dz�>���@�Q�BZ�
@�@E�>��R@�  B[33                                    BxW,��  �          @s�
?�@Mp�?�RA\)Blff?�@N{?
=qA�\Bl�H                                    BxW,�.  �          @r�\?ٙ�@N�R?fffA\z�ByG�?ٙ�@P  ?O\)AG�By�                                    BxW,��  �          @q�?�p�@X��?333A,z�B��3?�p�@Z=q?�RA33B��f                                    BxW,�z  �          @k�?�ff@R�\?�A�RB�u�?�ff@S33>�G�@�(�B���                                    BxW,�   T          @hQ�?��R@Vff?�\A��B�G�?��R@W
=>�(�@ٙ�B�ff                                    BxW-�  T          @h��?�z�@Z=q>��R@�33B�z�?�z�@Z�H>k�@dz�B��=                                    BxW-l  �          @l��?�
=@E>��@�(�Bg?�
=@Fff>���@�Q�Bh�                                    BxW-$  �          @l��?�33@HQ�>�@�RBj�H?�33@H��>Ǯ@ÅBk(�                                    BxW-2�  �          @k�?�=q@W�>�
=@ҏ\B�  ?�=q@XQ�>�{@��B��                                    BxW-A^  �          @c�
>��@`��>���@�B�� >��@`��>��
@�
=B��=                                    BxW-P  �          @g�?=p�@^�R?&ffA'
=B��H?=p�@_\)?z�AQ�B���                                    BxW-^�  �          @n�R?G�@c�
?E�A>�RB�B�?G�@dz�?333A,��B�\)                                    BxW-mP  �          @k�?(��@e>��@���B�L�?(��@fff>���@ə�B�W
                                    BxW-{�  �          @e>8Q�@c�
>8Q�@<��B��>8Q�@dz�=�?�\)B��                                    BxW-��  �          @e?��@aG�<�>�(�B��{?��@aG��#�
�(�B��{                                    BxW-�B  �          @a�?��@]p�=�\)?���B��?��@]p�<#�
=�B��                                    BxW-��  �          @C33?Tz�@2�\?   Az�B��
?Tz�@333>�ffAp�B��                                    BxW-��  @          ?�33>��
?�>L��@��B�.>��
?�>.{@���B�8R                                    BxW-�4            @\)>�  @�H?�AA�B�(�>�  @�>�A333B�33                                    BxW-��  �          @Dz�=�\)@@��?�A�
B�8R=�\)@AG�>�A��B�8R                                    BxW-�  �          @N�R>#�
@L(�>�(�@��RB�L�>#�
@L(�>\@ۅB�Q�                                    BxW-�&  �          @S�
���
@P  >k�@��
B��R���
@P  >B�\@R�\B��R                                    BxW-��  �          @B�\>L��@AG�>��@��HB�B�>L��@AG�>W
=@��B�B�                                    BxW.r  �          ?�
=��  ?L�Ϳ&ff�(�B�(���  ?L�Ϳ(���  Bր                                     BxW.  �          @)���h��@�ÿ#�
�`Q�Bݞ��h��@Q�+��k33B�                                    BxW.+�  
<          @<�Ϳs33@1G�=�Q�?�B���s33@1G�=u?�Q�B���                                    BxW.:d  �          @,(��G�@"�\>��@��\B�B��G�@"�\>k�@��RB�=q                                    BxW.I
  �          @���#�
@�R>�p�AB�  �#�
@\)>�{A  B�                                      BxW.W�  T          ?�
=?333?�\)?&ffA�ffB�B�?333?У�?#�
A�(�B�p�                                    BxW.fV  �          ?����
�\�u�ffC��
���
���
�s33��ffC��
                                    BxW.t�  �          ?�p���=q��\)�s33�	G�C��H��=q���׿p���G�C���                                    BxW.��  �          ?��ÿ�ff�G����R�.�CX����ff�J=q���R�-ffCX��                                    BxW.�H  �          @�׿У׿k����R��CQp��У׿n{��p��G�CQ��                                    BxW.��  �          @{��\)��׿�
=�!33CDG���\)����� �RCD�)                                    BxW.��  �          @(���  �k����R�2
=C<�Ϳ�  �u���R�1�
C<�f                                    BxW.�:  �          ?�  �}p���
=��p��Cz�CKG��}p���(���p��B��CK�H                                    BxW.��  �          @{��
=��  ��33�B33C=�q��
=�����33�B  C>T{                                    BxW.ۆ  
�          @���z�=�G���33�T�
C/����z�=��Ϳ�33�T�C/�H                                    BxW.�,  
�          @녿�
=�\)��33�h  C:�ÿ�
=�����33�g�HC;�                                    BxW.��  �          @33��z������ap�CJW
��z������a�CJ�)                                    BxW/x  �          @#33���R�aG��33�V{CWp����R�c�
�33�U�RCW�)                                    BxW/  �          @$z῕�G��	���c�CU�����G��	���b�HCU�                                    BxW/$�  �          @!녿�\)������p��.33C]Y���\)���H��p��-��C]h�                                    BxW/3j  �          @zῴz῁G��\�$Q�CW�{��z῁G���G��$33CW��                                    BxW/B  �          @   �޸R�aG���ff���CN���޸R�aG���ff���CN�                                    BxW/P�  �          @�H���
�5���H�Q�CI�R���
�5���H�\)CI�3                                    BxW/_\  �          @=q��=q�����
=���CFE��=q�����
=��CF:�                                    BxW/n  �          @�
��33�zῼ(�� =qCGuÿ�33�zῼ(�� ffCGaH                                    BxW/|�  �          @����ÿ\)��(���RCE
=���ÿ�Ϳ�(���HCD�                                    BxW/�N  �          @녿�녾Ǯ��������C?�=��녾\������
=C?p�                                    BxW/��  T          @
=q��
=    �s33�иRC4\��
=    �s33�иRC3��                                    BxW/��  �          ?��R����    �J=q���C4  ����    �J=q���C3��                                    BxW/�@  �          ?�Q�޸R<#�
�Tz���ffC3���޸R<#�
�Tz���ffC3�\                                    BxW/��  
�          @ �׿��=��Ϳ&ff����C1����=��Ϳ&ff���HC1                                    BxW/Ԍ  �          @�
��\�k����\��p�C:�=��\�k����\��C:^�                                    BxW/�2  �          @,����\�:�H������RCGǮ��\�8Q��{�=qCG��                                    BxW/��  T          @'
=��\)�L�ͿУ��G�CK���\)�G�������CJ��                                    BxW0 ~  �          @%����ͿJ=q�����CK.���ͿG���{�z�CJ�
                                    BxW0$  �          @-p���G������G��"33CS��G������\�#=qCR�3                                    BxW0�  �          @*=q�����=q��(�� z�C\޸������ÿ�p��!�HC\�                                    BxW0,p  �          @6ff��33��������HC]c׿�33��
=�����ffC]�                                    BxW0;  �          @3�
��녿�{���#\)C[����녿�������$�C[B�                                    BxW0I�  �          @0�׿�ff���
����,Q�C[}q��ff��G���33�-��C[                                    BxW0Xb  T          @1G��\���
��
=�0�C\�\��  ��Q��2p�C[��                                    BxW0g  �          @3�
��zῦff�����%(�CZ{��zῢ�\��\)�&��CY�{                                    BxW0u�  T          @+����
���ÿ�  �#�C\�=���
��ff��\�%�C\G�                                    BxW0�T  T          @*�H��Q쿵��G��	Q�C\
��Q쿳33���
�ffC[��                                    BxW0��  �          @&ff��
=�����33� ffC_@ ��
=���ÿ��"�RC^�R                                    BxW0��  �          @%��G�����=q�G�Cn�q��G���33��{�G�Cn�)                                    BxW0�F  A          @(Q�}p�����H�/33Cw
�}p������<��Cw                                    BxW0��  g          @*=q�s33��H���6ffCx� �s33�=q�\)�D��Cxk�                                    BxW0͒  �          @+��^�R�\)�����
=Cz�3�^�R��R��ff�{Cz��                                    BxW0�8  �          @-p��k�� �׾����Cy녿k��   ���
=Cy޸                                    BxW0��  �          @.{�z��'���z���(�C����z��'
=��{��p�C���                                    BxW0��              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxW1*   w          @aG��HQ�?u��ff���C"�q�HQ�?z�H���
��z�C"�\                                    BxW1�  �          @Z=q�@��?�  ���R����C!���@��?��
��(����C!!H                                    BxW1%v  �          @X���8��?��H�����33CW
�8��?�p����
��\)C޸                                    BxW14  �          @U��2�\?�G���ff��ffC���2�\?��
���\��=qCQ�                                    BxW1B�  �          @Q��5�?��Ϳ��R���C�{�5�?�\)���H���CW
                                    BxW1Qh  �          @XQ��8Q�?�(����
����C��8Q�?�  ��G����\Cp�                                    BxW1`  �          @U�3�
?��\���
���HC� �3�
?�ff��  ��Q�C:�                                    BxW1n�  �          @S�
�4z�?��\��
=��z�C���4z�?�ff������
CT{                                    BxW1}Z  �          @Vff�7
=?��\��Q���(�C���7
=?�ff��33��\)C}q                                    BxW1�   �          @Tz��1�?��׿����C��1�?�zῑ���Q�C                                    BxW1��  �          @S�
�.�R?��ÿ��
��z�C!H�.�R?˅�z�H��Q�C�3                                    BxW1�L  �          @\(��6ff?�  ����\)C:��6ff?��
������p�C��                                   BxW1��  �          @=q���H?�G�?��A�p�Ch����H?z�H?�{A��C.                                    BxW1Ƙ  �          @{���
?��=�Q�@ffC����
?��>�@Tz�C.                                    BxW1�>  �          @�
=q?8Q�>�(�A*�RC!� �
=q?5>�ffA333C!�\                                    BxW1��  �          @  ��Q�>��?�G�A�\)C,s3��Q�>k�?�G�A�
=C-W
                                    BxW1�  �          @z���\?=p�?8Q�A��HC�3��\?8Q�?=p�A��C ��                                    BxW20  �          @�
��\?��
=�@:�HC(���\?��
>��@n�RCB�                                    BxW2�  
�          @���ff?�33?   A=�C\)�ff?���?
=qAK33C�R                                    BxW2|  �          @.�R���?��>#�
@aG�C�\���?��>L��@��\C��                                    BxW2-"  �          @H���3�
?��\�z��*�HC�H�3�
?���
=q�{CO\                                    BxW2;�  �          @E�1�?�=q��\)����Cn�1�?���u��p�CG�                                    BxW2Jn  �          @C�
�5�?�\)��=q���CaH�5�?��׾u��p�C8R                                    BxW2Y  �          @7
=�+�?xQ�u��33C 
=�+�?xQ켣�
��ffC �                                    BxW2g�  �          @:=q�1G�?Y�����&ffC"�H�1G�?\(����Ϳ��RC"��                                    BxW2v`  �          @8Q��1�?+��aG���33C&k��1�?.{�B�\�uC&G�                                    BxW2�  �          @5��-p�?=p��u���
C$��-p�?@  �W
=��G�C$��                                    BxW2��  �          @5��(��?O\)���9G�C"�q�(��?Tz���.�HC"�\                                    BxW2�R  �          @<���+�?=p��z�H����C$��+�?G��s33���C#�                                    BxW2��  �          @@���1G�?c�
�B�\�j�\C"#��1G�?k��8Q��_33C!�
                                    BxW2��  �          @G
=�:=q?�  ��
=���C!!H�:=q?�G��\���C ٚ                                    BxW2�D  
�          @J=q�7
=?���Ǯ��C�3�7
=?�ff��{��(�Cs3                                    BxW2��  �          @QG��7�?\��  ���RC��7�?��
�B�\�U�C�3                                    BxW2�  �          @Vff�333?�<#�
>B�\C���333?�=�Q�?��
C�3                                    BxW2�6  �          @Tz��6ff?��H���	��C{�6ff?��H�L�Ϳ^�RC                                    BxW3�  �          @S33�8Q�?��;8Q��K�C�q�8Q�?��ͽ�G�����C��                                    BxW3�  �          @I���)��?�
==��
?�\)C���)��?�
=>��@0  C�3                                    BxW3&(  �          @G
=�p�?�녽�Q��p�C��p�?�논#�
�\)C}q                                    BxW34�  �          @G����?�p��#�
�O\)C�����?�p�=L��?aG�C�                                    BxW3Ct  �          @C33��?���>.{@O\)C���?�Q�>��@�
=C8R                                    BxW3R  �          @E���R@
=>�z�@�C� ��R@ff>Ǯ@��C�R                                    BxW3`�  �          @H���G�@�>�p�@�\)C	  �G�@ff>��A	�C	E                                    BxW3of  �          @?\)�
=?�  =�@�
Cp��
=?޸R>L��@\)C�\                                    BxW3~  �          @<(��Q�?�z�>\@���C
{�Q�?��>�A��C
ff                                    BxW3��            @3�
��@ ��?�G�A��C�Ϳ�?��H?�{A�33C�\                                    BxW3�X  �          @5���\?��R?h��A�ffC�R��\?�Q�?�G�A��
Cn                                    BxW3��  �          @4z���R?�33?!G�AN{CW
���R?�{?:�HAn�RC�H                                    BxW3��  �          @3�
��p�?�\)?0��AdQ�C���p�?�=q?J=qA��RC	�                                    BxW3�J  �          @5��33?�
=?Tz�A��Cz��33?��?n{A��HC0�                                    BxW3��  �          @
�H����?У�?O\)A�B��H����?�=q?c�
A���B�u�                                    BxW3�  �          @=q@���  ?z�A`z�C�!H@����
?�AJffC�˅                                    BxW3�<  �          @{@33�W
=>.{@w
=C��@33�Y��=�@0��C��{                                    BxW4�  �          @�H@ff�   >�@G�C��{@ff��\=���@(�C��H                                    BxW4�  �          @(�@�
�.{���
���C���@�
�+���G��-p�C��q                                    BxW4.  �          @�@
�H�p��=u?�33C�L�@
�H�p��    =��
C�G�                                    BxW4-�  �          @�@=q�\)<#�
>�=qC�P�@=q�\)    =��
C�O\                                    BxW4<z  S          @�H@�H<��
=u?��R>�(�@�H<#�
=u?�G�>�33                                    BxW4K   
�          @�R@p�=���<��
>Ǯ@G�@p�=���<��
?�@\)                                    BxW4Y�  �          @�R@(����
�#�
�p  C�5�@(����R�8Q���ffC�T{                                    BxW4hl  �          @#33@z῀  =���@�
C�^�@z῀  =#�
?k�C�S3                                    BxW4w  �          @�?�p����R>��A5�C�  ?�p����\>ǮA�C���                                    BxW4��  �          @��?��
���\>��AMp�C�
=?��
��ff>���A0z�C��
                                    BxW4�^  �          @=q@  ��(����K�
C���@  ���Ϳ
=q�VffC��)                                    BxW4�  �          @ ��@��\)�^�R��C�B�@���Q�aG���33C���                                    BxW4��  �          @�?�(�>�����
=���A
�H?�(�>\��z����A-G�                                    BxW4�P  �          @\)?�?��ÿ�=q�33B���?�?�녿}p��(�B��q                                    BxW4��  �          @0  ��ff@G���  ��  B�8R��ff@��^�R��B��=                                    BxW4ݜ  �          @:=q��@
�H�p�����RB�G���@�R�J=q�|z�B���                                    BxW4�B  �          @4z��  @ff������=qB�.��  @
�H�n{����B�u�                                    BxW4��  T          @6ff����@G����H��=qB�׿���@ff�����ffB�\                                    BxW5	�  �          @@�׿c�
@z��33�\)B��H�c�
@���p���
=B�(�                                    BxW54  �          @K��W
=@#�
�˅��ffB�B��W
=@*�H��z����
B��                                    BxW5&�  �          @E��(�?�\)���Q�Cff�(�?�33��Q���Q�C�                                    BxW55�  �          @U�7
=?�=u?�  C���7
=?�z�>8Q�@C�
C�\                                    BxW5D&  �          @Vff�B�\?��>��Ap�C  �B�\?�  ?\)A
=C��                                    BxW5R�  �          @Vff�Dz�?�33?#�
A0z�Cz��Dz�?���?8Q�AG�C L�                                    BxW5ar  �          @]p��AG�?���>��@ڏ\C��AG�?��?�A(�C
=                                    BxW5p  �          @HQ��2�\?�G�?�AC�)�2�\?�(�?�RA9p�CY�                                    BxW5~�  �          @hQ��E�?�33>u@uC^��E�?��>\@\C��                                    BxW5�d  �          @h���C33?�p�<��
>���C)�C33?�(�>.{@)��C5�                                    BxW5�
  �          @c33�1�@(��W
=�X��C�q�1�@�ͽ#�
�+�C��                                    BxW5��  �          @_\)�   @����H�{C� �   @=q���R��=qC!H                                    BxW5�V  
�          @\(��{@z�\)���C�\�{@
=�\����C^�                                    BxW5��  �          @X���{@�׿�\�
�\C	� �{@33������33C	{                                    BxW5֢  �          @Y���!G�@�R���H���C
�\�!G�@�׾��R����C
(�                                    BxW5�H  �          @Q���@
=q��G���z�C
T{��@(������=qC	��                                    BxW5��  �          @Q녿�Q�@��������C���Q�@{�\(��up�C �                                    BxW6�  �          @P  �
=q?�(��h����{C	���
=q@녿=p��aG�C��                                    BxW6:  �          @R�\�333?��
?@  ATz�Cff�333?��H?^�RAx  CxR                                    BxW6�  �          @U�.�R?�?(�A*=qCff�.�R?�  ?B�\AS�
C@                                     BxW6.�  �          @Q��+�?�\?�AC�\�+�?�(�?.{A?�CQ�                                    BxW6=,  �          @Tz��*�H?�
=>��
@�z�C��*�H?�33>��HA\)C�
                                    BxW6K�  �          @Tz��'�?�(�>�ff@�Q�C��'�?�
=?(�A*�HC��                                    BxW6Zx  �          @Tz��-p�?��?0��AA�C���-p�?�(�?Y��Al  C�
                                    BxW6i  �          @U�(Q�?�z�?@  APQ�C��(Q�?�?h��A~�\C�R                                    BxW6w�  
�          @H���#33?���?�Q�A�  CL��#33?p��?��
A��C��                                    BxW6�j  f          @J�H�=q?�  ?��A�Q�C��=q?�\)?�G�A���CW
                                    BxW6�  �          @J=q�#33?�  ?��A�z�C��#33?���?���A�C\                                    BxW6��  �          @I���(Q�?��?��
A£�Cn�(Q�?k�?�\)A���C �3                                    BxW6�\  �          @P�׿У�@>�  @�\)B�Ǯ�У�@�
>�G�A��C O\                                    BxW6�  �          @X�ÿ��R@G
=��
=��
=B߮���R@HQ�\)��B�W
                                    BxW6Ϩ  �          @U��z�@�#�
�J=qC���z�@>��@<��C                                    BxW6�N  �          @Tz��%�?\?���A�  C�\�%�?���?\A���C��                                    BxW6��  �          @W��"�\?�{?�A�p�Cٚ�"�\?޸R?��A�(�C�)                                    BxW6��  �          @Vff�'
=?�Q�?333AC\)Cz��'
=?�{?aG�Au�C��                                    BxW7
@  �          @`����@&ff?=p�AE��C���@ ��?z�HA��
C�R                                    BxW7�  �          @b�\�#33@=q>�z�@���C���#33@�?�AQ�C	\                                    BxW7'�  �          @a��(�@.�R>�{@��C �R�(�@,(�?��A��C0�                                    BxW762  �          @e��=q@C�
��\)��G�B�3��=q@Dz�<#�
>k�B�z�                                    BxW7D�  �          @c33��Q�@L�;���=qB�G���Q�@L��>.{@,��B�G�                                    BxW7S~  �          @_\)�Q�@W
=��z����
B�Q�Q�@XQ�=#�
?(�B�8R                                    BxW7b$  �          @e��\(�@[�����\B�33�\(�@^{�k��i��B��f                                    BxW7p�  �          @e���(�@[��Y���[�B�G���(�@`  �   � ��B�                                    BxW7p  �          @hQ쿵@R�\>k�@p  B���@P  ?\)Az�B�33                                    BxW7�  �          @j�H��Q�@L(�?!G�A{B�녿�Q�@G
=?s33Ao�
B�#�                                    BxW7��  �          @k��\@U>.{@%�B�\�\@S�
?   @�z�B�                                     BxW7�b  �          @n{��@QG����H��z�B�=q��@S33�#�
���B�                                    BxW7�  �          @h���  @,(�?xQ�Aw\)C�f�  @%�?��RA�C!H                                    BxW7Ȯ  �          @n{��G�@L(�?
=A�RB��H��G�@G
=?k�Af{B��                                    BxW7�T  �          @n{���@?\)>\@�{B�
=���@<(�?0��A-�B�{                                    BxW7��  �          @l�Ϳ��R@C�
?�A{B�.���R@?\)?W
=AS
=B�k�                                    BxW7��  �          @hQ��!�@p�?@  A@��C�{�!�@
=?�G�A�{C�3                                    BxW8F  �          @n�R��G�@\��>aG�@\(�B�  ��G�@Z=q?
=A\)B�z�                                    BxW8�  �          @r�\��G�@j=q���
�k�B��쿁G�@h��>\@�\)B���                                    BxW8 �  �          @p  ��=q@dz����BՊ=��=q@dz�>u@g�BՔ{                                    BxW8/8  �          @q녿�(�@c�
��=q���HB�𤿜(�@dz�=�?�=qB��)                                    BxW8=�  �          @q녿�G�@^{�5�.�RB��쿡G�@a녾������B�8R                                    BxW8L�  �          @o\)��(�@U����{B�=��(�@XQ�k��c�
B���                                    BxW8[*  �          @n{�z�@>{>�(�@ָRB����z�@9��?@  A>�HB��)                                    BxW8i�  �          @n�R�\)@)��?W
=AO�C.�\)@"�\?���A���Cff                                    BxW8xv  �          @p  ��
@;�>�z�@�p�C :���
@8Q�?�RA�C ��                                    BxW8�  �          @n�R�33@A�?.{A'�
B�\�33@<(�?��\A}B���                                    BxW8��  �          @o\)��z�@A�?��A��B�aH��z�@8��?�33A�  B��)                                    BxW8�h  �          @p�׿�p�@,(�@�B��B����p�@�H@�RB)ffB���                                    BxW8�  �          @qG���  @,(�@�B!{Bܣ׿�  @��@.�RB9B�=q                                    BxW8��  �          @r�\��z�@AG�?���A��\B���z�@1�@�\B{B�ff                                    BxW8�Z  �          @p  �z�H@J=q?�  A�\)B֊=�z�H@;�@
=B	=qB�                                    BxW8�   �          @i����G�@<��?�p�A��HB�=q��G�@.{@z�B
(�B噚                                    BxW8��  �          @hQ쿝p�@	��@&ffB9B���p�?�=q@5BP�B��f                                    BxW8�L  �          @c�
��@@\)B(�B���@33@   B4�B�G�                                    BxW9
�  �          @g���z�@-p�?�z�A���B�33��z�@\)?�(�B��B��{                                    BxW9�  �          @g�����@.�R?�G�A�=qB��Ϳ���@   @z�B�RB�W
                                    BxW9(>  �          @dz���R@5�?.{A1�B�LͿ��R@.�R?��\A��
B�L�                                    BxW96�  �          @c�
�   @1G�?8Q�A=�B��
�   @*=q?�ffA��B�                                    BxW9E�  T          @a���
@1G�>�@�z�B�#���
@,(�?O\)AXQ�B�                                    BxW9T0  T          @fff���R@Q녿z���RB�\)���R@U��.{�1�B��
                                    BxW9b�  �          @b�\�˅@G�>��@�B��)�˅@C33?J=qAPQ�B�                                      BxW9q|  �          @^�R��
=@@��>��@�Q�B�G���
=@<��?!G�A(Q�B�.                                    BxW9�"  �          @`  ��@8Q���	�B�LͿ�@;�����   B�\                                    BxW9��  �          @^{���@&ff>\)@\)C����@#�
>�@�ffCk�                                    BxW9�n  �          @aG����@2�\<�?�B����@0��>Ǯ@�G�B��\                                    BxW9�  �          @g
=��@:�H�������B��3��@;�=��
?��
B�p�                                    BxW9��  �          @c�
�   @3�
���z�B����   @7
=����   B���                                    BxW9�`  �          @`�׿��H@8�þ�z�����B�(����H@:=q=��
?���B��                                    BxW9�  �          @Y������@#33�:�H�Lz�B������@'��Ǯ�ڏ\B�B�                                    BxW9�  �          @\(����@�\?�(�A���C�
���?�{?�p�AυC!H                                    BxW9�R  �          @H�ÿ���@z�>��
@ۅB�8R����@��?�RAUB�{                                    BxW:�  �          @?\)���@p��u���C�����@{=L��?uCp�                                    BxW:�  �          @>�R��?�Q�?�\A1��C޸��?�\)?+�Ag�Ck�                                    BxW:!D  �          @C�
��H?�?��RA��
C&�
��H>�Q�?ǮB�RC+}q                                    BxW:/�  �          @A��0  ���k���{C>  �0  ��ff��z���\)C=J=                                    BxW:>�  �          @G��ff?��
�����p�Cu��ff?�(�������{C��                                    BxW:M6  �          @L(��W
=?�ff�'��n{B��R�W
=?����H�S{B��                                    BxW:[�  �          @C�
�:�H?�>�G�A�C)��:�H>�ff?�\A(�C+(�                                    BxW:j�  �          @L(��0��?�(��B�\�`��C(��0��?���z��,z�C�{                                    BxW:y(  �          @W��\)?\�����C��\)?޸R������  C�                                    BxW:��  �          @L�Ϳ��H?��H����=qC�3���H?�Q�����{CT{                                    BxW:�t  �          @C�
�Q�@���\)���B�8R�Q�@$zΰ�
�ʏ\B�ff                                    BxW:�  �          @L�Ϳ}p�@ff���)��B�8R�}p�@Q���=qB�                                    BxW:��  �          @O\)>8Q�?�33�/\)\B��q>8Q�?\�#33�l(�B�(�                                    BxW:�f  �          @c33?��H?ٙ���(33B#��?��H@ ���z���B6��                                    BxW:�  
�          @7���33@�
���H��33B�(���33@p��h�����HB��H                                    BxW:߲  �          @8�ÿ�ff@녿��H���HB��{��ff@��h����{B���                                    BxW:�X  �          @�H�Y��?�G��У��,z�B��)�Y��?�p���33�ffB�                                    BxW:��  �          @b�\��{@
=q�1G��N��B��þ�{@!����.=qB�W
                                    BxW;�  �          @e=��
@���)���?��B�33=��
@0  �����B��3                                    BxW;J  �          @Z�H=���@-p���
��RB��)=���@>�R�����RB�8R                                    BxW;(�  �          @^�R>#�
@:�H��\)�\)B��{>#�
@J=q����B�\                                    BxW;7�  �          @Z=q>�{@@  ��  ��p�B�  >�{@L(���ff��z�B�                                    BxW;F<  �          @Z�H>��@4z��33�B��>��@Dzῼ(����
B��\                                    BxW;T�  �          @fff?�p���33�:�H�]��C���?�p�=�Q��<(��`=q@�R                                    BxW;c�  �          @c�
@   �#�
�-p��U�C���@   >�  �,���U(�@���                                    BxW;r.  �          @i��@9���E����R�\)C��@9���������\C�h�                                    BxW;��  T          @j�H@3�
�����
���C��@3�
�Q�����  C��)                                    BxW;�z  �          @l��@8�ÿ�������C�޸@8�ÿ@  �
�H��HC��)                                    BxW;�   �          @[�@
=�녿�
=�%�RC�p�@
=��=q���R�,�\C�T{                                    BxW;��  �          @[�?B�\?#�
�P  �)B ��?B�\?����G
=�RB_�                                    BxW;�l  C          @`��?h��>��:�HG�A���?h��?k��4z��RB4��                                    BxW;�  �          @x��@;����{�\)C���@;��8Q��!G��"�
C�7
                                    BxW;ظ  �          @y��@Fff�#�
�G��z�C�33@Fff��\)���C�k�                                    BxW;�^  �          @z=q@N�R����
=�(�C�0�@N�R�W
=�
�H�Q�C��                                    BxW;�  �          @qG�@<�;�Q���\�33C��@<��    �z��ffC��q                                    BxW<�  �          @p��@7��Ǯ����\)C�R@7����
��� �HC��R                                    BxW<P  �          @g�@,(�>��
=�%33@7
=@,(�?   ��
� ��A)��                                    BxW<!�  �          @dz�@6ff>��H��
�A33@6ff?O\)���H�A|Q�                                    BxW<0�  T          @fff@8��>�����
@%�@8��>��z���A                                    BxW<?B  C          @h��@3�
?z��
=q�A<  @3�
?k���\�
Q�A���                                    BxW<M�  e          @^�R@(Q�<#�
�\)�!�
>�\)@(Q�>�p��p��(�A (�                                    BxW<\�  �          @^�R@-p�?����=q��A���@-p�?�\)�У���\)Aՙ�                                    BxW<k4  �          @b�\@8��?aG��ٙ���\A���@8��?�녿��
��=qA��
                                    BxW<y�  �          @e�@9��?����\��Q�A�\)@9��?�{������G�A��
                                    BxW<��  �          @c33@0��>�
=�
=q��A��@0��?B�\�z��(�Av{                                    BxW<�&  �          @N�R@1G�?   ���R�߮A#�@1G�?:�H�����\)Am�                                    BxW<��  �          @7
=?�  >\�����JA��R?�  ?#�
�\�<�A��H                                    BxW<�r  �          @'��\���\��R��RCy�q�\�#�
��  Co�)                                    BxW<�  �          @@��?\�B�\����e��C���?\>W
=����eQ�@�z�                                    BxW<Ѿ  �          @XQ�@
=?G��\)�(
=A�(�@
=?������RA�                                      BxW<�d  �          @<��?�  ���ff�S�\C�|)?�  �#�
�	���[��C��=                                    BxW<�
  T          @Q�@�H>#�
�
�H�'{@l(�@�H?��
=�!�
A@                                      BxW<��  T          @Dz�@(�>����
�z�A+�@(�?@  ��
=�33A�33                                    BxW=V  �          @`  @5?8Q��\)�=qAdz�@5?��
��(���A�
=                                    BxW=�  �          @aG�@+�?Tz��ff�=qA�{@+�?�
=��
=���A�\)                                    BxW=)�  �          @e@7
=?aG����H��A��\@7
=?��������G�A�                                    BxW=8H  �          @e�@7
=?Tz��p����A�{@7
=?�z����p�A�ff                                    BxW=F�  �          @P��@%?E�����{A��@%?���У���  A�(�                                    BxW=U�  �          @���@7
=���^{�I�C�˅@7
=>���\(��G=qA�\                                    BxW=d:  �          @��@
=q�:�H�{��oz�C���@
=q�#�
��  �vQ�C�y�                                    BxW=r�  �          @���@ff�!G��}p��t(�C��)@ff=u��Q��yz�?�{                                    BxW=��  �          @���@
=q�8Q������r�RC���@
=q�#�
���
�y\)C���                                    BxW=�,  �          @��@33�=p�����xz�C��@33���
����C�Ǯ                                    BxW=��  �          @���@   �������}�C���@   >���ff��@u                                    BxW=�x  e          @��@33��
=����}(�C�>�@33>��
��p��~(�AQ�                                    BxW=�  �          @�33@
=�L�����H�zp�C�O\@
=?�����w\)A^ff                                    BxW=��  �          @��
@ ��>�
=�w
=�bp�Ap�@ ��?����n{�VA�
=                                    BxW=�j  �          @��
@%�>��r�\�](�A)G�@%�?���h���Q  A�\)                                    BxW=�  �          @r�\?c�
���<(��Q�C���?c�
���
�QG��v=qC�1�                                    BxW=��  �          @{�?��H����G
=�RQ�C��=?��H��ff�Y���pp�C�33                                    BxW>\  �          @r�\?����R�H���b��C�� ?��fff�Vff�}=qC�Ф                                    BxW>  �          @x��@�ÿ���7
=�=z�C���@�ÿ\)�@���J�C�ff                                    BxW>"�  �          @~{?��.{�W
=�k  C�T{?��u�Z�H�r�RC��                                    BxW>1N  �          @�ff@���ff�_\)�`��C��R@�����g��np�C�|)                                    BxW>?�  �          @��?�=q�8Q��|��\)C�@ ?�=q?
=q�z=q��A���                                    BxW>N�  �          @}p�?���>�\)�p����APQ�?���?u�i��k�Bff                                    BxW>]@  
�          @�{@�\�#�
�j�H�s��C�aH@�\?!G��g
=�mA�Q�                                    BxW>k�  T          @�G�@�
>�(��c�
�b33A)��@�
?�=q�Z�H�U(�A�                                    BxW>z�  �          @�ff?��=#�
�����f?��R?��?G��|(��z�A�                                      BxW>�2  �          @�G�?��<#�
�vff�z�>�{?��?8Q��q��w�A���                                    BxW>��  �          @�  @ ��?���Z=q�j�At��@ ��?��P  �Y�
A��
                                    BxW>�~  �          @y��?��H?G��Q��eG�A���?��H?�\)�E��P��BG�                                    BxW>�$  �          @l��?�z�?����Dz��c=qB	  ?�z�?�
=�3�
�G��B5z�                                    BxW>��  �          @tz�?�?&ff�N{�k=qA���?�?��R�B�\�W��Bz�                                    BxW>�p  �          @vff?޸R��
=�Y���x�C�'�?޸R>aG��Z�H�{33@�p�                                    BxW>�  �          @{�@
=q?L���L(��X�RA�ff@
=q?����>�R�EG�B��                                    BxW>�  �          @{�@33>����HQ��U��A@33?h���@���J�A��                                    BxW>�b  �          @vff?�=q���W
=�uz�C���?�=q?(��S33�n�A�33                                    BxW?  T          @\)?�ff���`���v�HC�&f?�ff=����c33�|z�@O\)                                    BxW?�  
�          @\)?��H�p���^�R�rz�C��
?��H��=q�ep�C���                                    BxW?*T  �          @r�\?����p��HQ��^�C�?���\(��Vff�yffC�n                                    BxW?8�  �          @h��?��H��{�B�\�b�C�� ?��H�B�\�O\)�|ffC�S3                                    BxW?G�  �          @U�?��\��=q�0  �vz�C�?��\���:=q�qC�Q�                                    BxW?VF  �          @녿��?�(�?�=qB3(�B�G����?��?�=qB[�B�k�                                    BxW?d�  �          ?�=q�.{?�Q�>#�
@�G�B���.{?У�>�A���B���                                    BxW?s�  �          ?�G����
?����Ǯ���RBѸR���
?�G��������BЊ=                                    BxW?�8  �          ?�Q�=L��?�=q�Y���(�B���=L��?�p��!G�����B��\                                    BxW?��  �          ?ٙ���G�?�33�G���RB�B���G�?\���H���B��                                     BxW?��  �          ?�논#�
��\)��(��SQ�C��ͼ#�
�O\)��z�33C��R                                    BxW?�*  �          ?�\�����H<#�
>�CvY�����Q�k��{Cu�R                                    BxW?��  �          ?�(����
�#�
?�=qBb{C<�׿��
���?\BT�CLn                                    BxW?�v  T          @p���?�=q>L��A#�B��{��?��\>��A���B�Ǯ                                    BxW?�  �          @
�H>�(�?��^�R����B��)>�(�@33����J�HB�p�                                    BxW?��  �          @+�?�@(��n{��
=B�\?�@$z��
=��HB�G�                                    BxW?�h  �          @@��>Ǯ@.�R��(���ffB���>Ǯ@:�H�&ff�F�HB��
                                    BxW@  �          @W�=�\)@@  ��
=�̏\B�\)=�\)@N{�O\)�`��B��=                                    BxW@�  2          @Z=q?��@G����'�\B|�?��@+����
��\)B��R                                    BxW@#Z  "          @~{?�?����7
=�9\)B6  ?�@\)��H�Q�BQp�                                    BxW@2   �          @~�R?�\)?����:=q�<33B8��?�\)@   �����BT�                                    BxW@@�  
(          @�Q�?���@G��<(��<B?��?���@$z��{�Q�BZ�                                    BxW@OL  
Z          @�(�?�G�@z��E�B��BFp�?�G�@*=q�'
=�33Ba�                                    BxW@]�  T          @��?�z�?���W��X\)B<��?�z�@��;��3��B^�H                                    BxW@l�  
�          @��?�  ?�
=�Mp��K�HB?=q?�  @"�\�0  �&�B]�H                                    BxW@{>  T          @�G�@{@
�H�#�
���B1z�@{@)����
���RBH�                                    BxW@��  
�          @���@�?�p��H���;(�B)=q@�@%��*�H�
=BGff                                    BxW@��  
�          @�Q�@33?��Dz��9�\B=q@33@=q�(����\B9Q�                                    BxW@�0  
�          @��@�@ ���7��,(�B#=q@�@#�
����
�B>                                    BxW@��  
�          @��\@=q?�Q��K��<�RB�@=q@�
�0���G�B/\)                                    BxW@�|  �          @�33@G�?�{�Mp��>33B�@G�@�R�0  �33B>�\                                    BxW@�"  "          @��R@��@���9���/33B3�H@��@,������ffBN{                                    BxW@��  �          @���?��R@�1��/��B9�?��R@'����
=BR                                    BxW@�n  
�          @w�?�p�?�(��>�R�I�BT(�?�p�@"�\� ���!�RBoG�                                    BxW@�  
�          @w
=?�\)?��R�8Q��A��BKz�?�\)@"�\�=q�ffBfQ�                                    BxWA�  �          @|��?�Q�?��H�H���R(�B5G�?�Q�@��.{�-(�BX{                                    BxWA`  "          @��
@(�?���U�G�
A�Q�@(�@�
�>�R�,  B G�                                    BxWA+  
�          @���@�
?Ǯ�7
=�7  Bz�@�
@Q��{��B*�H                                    BxWA9�  
�          @�z�@p�?��?\)�9G�BQ�@p�@���"�\��B=z�                                    BxWAHR  �          @��
@%?��
�E�7Q�A�z�@%@	���,���
=B                                    BxWAV�  T          @��\@#33?��H�QG��D�
A���@#33?�\)�<(��+�\B=q                                    BxWAe�  �          @�p�@-p�?�\)�R�\�AQ�A��@-p�?���>�R�*(�B��                                    BxWAtD  �          @��
@-p�?\(��U��F
=A��@-p�?�ff�Dz��2
=A���                                    BxWA��  
�          @��@{?&ff�W��SQ�Ak�
@{?�{�J=q�@�
A�                                    BxWA��  �          @��H@)��?L���U��H��A�(�@)��?��R�E��5ffA�
=                                    BxWA�6  �          @�{@,��?J=q�\(��J�HA���@,��?�G��L(��7�A��
                                    BxWA��  v          @��@!�?J=q�]p��R(�A��\@!�?�G��Mp��=��A�33                                    BxWA��  
�          @��\@\)?�R�^�R�Vz�A^�R@\)?����P���D=qA�(�                                    BxWA�(  2          @z=q@�?J=q�H���U��A�@�?����8���?=qB�\                                    BxWA��  
�          @z�H?�33>��XQ��op�Ac�?�33?����L���[\)B33                                    BxWA�t  	�          @�  ?�z�?�Q��hQ��kG�B#?�z�@(��N�R�FffBS�                                    BxWA�  
�          @��?�z�?�{�a��g��A�R?�z�?����L���I33B0(�                                    BxWB�  
�          @�Q�@p�?L���N{�WG�A�\)@p�?�p��>{�@G�Bff                                    BxWBf  "          @��H@33?��H�R�\�X=qA�  @33?�33�<(��9�
B+
=                                    BxWB$  T          @�(�?�p�?}p��p���n=qA�ff?�p�?���]p��Q\)B(z�                                    BxWB2�  T          @�ff@�?!G��y���u�\A���@�?�(��j�H�^��B=q                                    BxWBAX  
�          @{�?��?p���O\)�d=qAՙ�?��?���=p��G��B$(�                                    BxWBO�  "          @�
=@��?(��Y���c(�A�  @��?����K��N{B �
                                    BxWB^�  
�          @��@5����Z�H�G(�C���@5>aG��]p��J{@�ff                                    BxWBmJ  �          @���@Dz��G��I���6��C���@Dz�?
=�E�2��A.ff                                    BxWB{�  T          @��@6ff�aG��E�7��C�c�@6ff�8Q��Mp��AG�C�:�                                    BxWB��  
�          @j�H?������5��M33C�?��
=q�A��b=qC�%                                    BxWB�<  T          @Vff?�G����R�!G��E�
C�g�?�G��&ff�.�R�^{C��                                    BxWB��  T          @<(�?�G��p����
�QG�C�H?�G�����p��f(�C��H                                    BxWB��  
�          @=p�?��H�Q���H�]33C�ff?��H����"�\�n�
C��                                    BxWB�.  T          @^{?��ÿJ=q�5��[�HC�8R?��þ\)�<(��h��C��{                                    BxWB��  T          @S33?ٙ��.{�.�R�`�HC�  ?ٙ��L���3�
�k�\C�9�                                    BxWB�z  �          @]p�?�׿!G��4z��[�RC���?��<��
�8���c�?:�H                                    BxWB�   �          @W
=?^�R����B�\z�C��=?^�R��33�L��p�C�q                                    BxWB��  �          @Q�>�
=����?\)8RC�
>�
=�z��L��ǮC�{                                    BxWCl  �          @Mp�?
=q����<(�z�C���?
=q��ff�HQ���C�<)                                    BxWC  T          @Dz�?O\)���\���wffC�>�?O\)��G��&ff��C��                                     BxWC+�  T          @J�H?�녾.{�<��L�C��?��?�\�:=q��A�G�                                    BxWC:^  T          @J�H?��H���
�8���C��{?��H>�33�8��A�G�                                    BxWCI  "          @.{?E��u����zC�j=?E��Ǯ�#33k�C���                                    BxWCW�  �          @?\)?5�c�
�0  ��C�C�?5�u�8Q��
C���                                    BxWCfP  
�          @p  ?�ff?��HQ��q��BU�
?�ff@ff�.{�D�\B}p�                                    BxWCt�  T          @}p�?��H@Q��8���=(�B|Q�?��H@>�R�G��B��)                                    BxWC��  �          @���?�Q�@�R�:�H�8�
BS�R?�Q�@6ff����BmQ�                                    BxWC�B  T          @��?��H@
=q�>�R�<��BN33?��H@2�\����(�Bi��                                    BxWC��  �          @�?�@���?\)�6��BK33?�@8������\)Be�H                                    BxWC��  �          @��\?�(�@{�<���9ffBQ=q?�(�@6ff�ff��Bk�                                    BxWC�4  "          @���?޸R@��J�H�Gz�BE�H?޸R@-p��'
=�=qBeff                                    BxWC��  T          @��?��R?�
=�S33�V  BQ��?��R@)���0���)
=Br�\                                    BxWCۀ  T          @��\?�Q�?�G��X���`�BKp�?�Q�@   �8Q��3��Bp�                                    BxWC�&  �          @���?��
>����g��
AlQ�?��
?��R�[��p�B�
                                    BxWC��  �          @��?˅?J=q�g��G�A��
?˅?�\)�U��_
=B5�R                                    BxWDr  �          @k�?��\?�p��QG��BI�?��\?�(��9���R\)Bz(�                                    BxWD  �          @]p�?�  ?+��O\)p�B(�?�  ?�z��@  �pQ�BZ�H                                    BxWD$�  
�          @s�
?�G��k��i��L�C��\?�G�?!G��fffffB (�                                    BxWD3d  �          @���?�G��u�x���
C�0�?�G�?Y���s33��B �                                    BxWDB
  �          @���?���?���}p�u�A���?���?\�mp��}ffB[
=                                    BxWDP�  �          @�Q�?�  >Ǯ�vff�fA�ff?�  ?��
�j=q��BP33                                    BxWD_V  �          @�=q?��\=L���|(��H@&ff?��\?z�H�s�
\)B.�                                    BxWDm�  
�          @z�H?L�;k��tz��=C��?L��?.{�qG��
B"��                                    BxWD|�  �          @w
=?(�þu�r�\��C��H?(��?(���o\)�B4z�                                    BxWD�H  
�          @tz�?Tzᾔz��n{k�C�` ?Tz�?���k��HB�H                                    BxWD��  �          @r�\>�33���n�R¡  C�Ф>�33>\�p  ¤Q�B?��                                    BxWD��  T          @�  ?�녾�p��r�\��C�3?��?���qG��A�p�                                    BxWD�:  �          @qG�?��
��33�X��B�C�}q?��
>���XQ�L�A�33                                    BxWD��  �          @W�?�Q쾣�
�@����C���?�Q�>���?\)�}�HA�
                                    BxWDԆ  �          @Z=q�#�
�
=q�   �D�\C��
�#�
���R�<(��{�RC��                                    BxWD�,  �          @U�\(��4z��{��C}��\(��ff�\)�'�Cy޸                                    BxWD��  
�          @U��:�H�A녿����z�C�7
�:�H�*�H��=q�{C~��                                    BxWE x  h          @p�׿W
=�P�׿����ǮC�=�W
=�1��33�z�C}5�                                    BxWE              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxWE�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxWE,j              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxWE;              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxWEI�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxWEX\              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxWEg              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxWEu�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxWE�N              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxWE��              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxWE��              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxWE�@              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxWE��              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxWE͌              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxWE�2              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxWE��              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxWE�~              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxWF$              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxWF�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxWF%p              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxWF4              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxWFB�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxWFQb              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxWF`              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxWFn�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxWF}T              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxWF��              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxWF��              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxWF�F              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxWF��              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxWFƒ              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxWF�8              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxWF��              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxWF�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxWG*              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxWG�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxWGv              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxWG-              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxWG;�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxWGJh              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxWGY              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxWGg�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxWGvZ              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxWG�               ��O���O���O���O���O���O���O���O���O���O���O�                                  BxWG��              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxWG�L              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxWG��              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxWG��              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxWG�>              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxWG��              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxWG�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxWG�0              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxWH�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxWH|              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxWH&"              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxWH4�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxWHCn              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxWHR              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxWH`�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxWHo`              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxWH~              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxWH��              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxWH�R              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxWH��              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxWH��              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxWH�D              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxWH��              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxWH�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxWH�6              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxWI�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxWI�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxWI(              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxWI-�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxWI<t              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxWIK              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxWIY�              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxWIhf              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxWIw              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxWI��              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxWI�X              ��O���O���O���O���O���O���O���O���O���O���O�                                  BxWI��  ^          @j=q@#33�@  �\)�,p�C�Ǯ@#33���
�&ff�6�C�'�                                    BxWI��  �          @p  ?�G�?�{�O\)�pB<ff?�G�@
=q�1G��?��Bn�R                                   BxWI�J  	�          @l��?��\?�33�J=q�lB?�?��\@
�H�+��;\)Bo�                                    BxWI��  
�          @l��?Y��?��G
=�eG�B��?Y��@#�
�!��,�RB�.                                    BxWIݖ  "          @i��?\(�?���@���]�RB�#�?\(�@'
=�=q�$��B���                                    BxWI�<  �          @e�?�\@Q��3�
�P33B�?�\@2�\�	�����B�B�                                    BxWI��  T          @fff=�G�@ff�.�R�D�HB�p�=�G�@>�R� ���{B���                                    BxWJ	�  
�          @fff��\)@(��(���=
=B�Lͽ�\)@C33��z�� (�B���                                    BxWJ.  �          @fff�W
=@'�����,=qB���W
=@J�H��
=����B��                                    BxWJ&�  �          @`  ��
=@,(������BŨ���
=@J�H�������B�                                    BxWJ5z  T          @\(����@'��
�H�33B��H���@E��33���B�(�                                    BxWJD   T          @N{��  @,�Ϳ�p��Q�B�uþ�  @C�
�u���B�=q                                    BxWJR�  T          @G���33@{��{��B�Q쾳33@7���33��33B�
=                                    BxWJal  �          @E���ff@p���G��G�BȞ���ff@5��ff��(�B��                                    BxWJp  �          @I�����R@%��  ��\B��\���R@<�Ϳ�G���B��H                                    BxWJ~�  "          @J�H��=q@(Q��(��p�B�����=q@?\)�xQ���33B�B�                                    BxWJ�^  �          @HQ�B�\@*=q�У���p�B���B�\@@  �^�R��p�B�8R                                    BxWJ�  T          @L�;�p�@*�H��(��
=B��)��p�@AG��s33����B�                                      BxWJ��  �          @Z�H�0��@L�Ϳfff�u�B�z�0��@Tzἣ�
��Q�B˙�                                    BxWJ�P  "          @`  �0��@N{������B�#׿0��@Z=q��\)���
B���                                    BxWJ��  
�          @W
=�p��@L�;L���`��BԨ��p��@H��?&ffA2�\B�=q                                    BxWJ֜  
�          @Z�H���\@J�H�@  �K33B׸R���\@P  =�G�?��B���                                    BxWJ�B  
�          @X�ÿz�H@K���
=����B�8R�z�H@J�H>�G�@�B�=q                                    BxWJ��  T          @\(�����@N�R��(���RB؅����@N�R>�G�@��HB؅                                    BxWK�  �          @\�Ϳc�
@S�
������ffB�33�c�
@Q�?\)Ap�B�z�                                    BxWK4  T          @Z=q��\)@�?޸RA�\)C����\)?��@  B(p�C
�                                    BxWK�  
�          @[���@p�?��
A��C
��?�z�@33B(ffC:�                                    BxWK.�  T          @\(����H@z�?�{A��C5ÿ��H?�@
=qB{C	B�                                    BxWK=&  �          @Z=q�33@33?�p�A͙�C�3�33?���@�B\)C
L�                                    BxWKK�  
�          @]p��   @%�?�{A�ffB�u��   @��?�  A���CJ=                                    BxWKZr  �          @`  ���@N{=L��?L��B߽q���@E?fffAp(�B�aH                                    BxWKi  "          @i�����@Z=q�:�H�8��B��f���@_\)>L��@I��B�=q                                    BxWKw�  
Z          @i���n{@\(��:�H�:=qB���n{@`��>L��@L(�Bх                                    BxWK�d  T          @`�׿��
@P��    ��B��f���
@H��?\(�Ac�B�Q�                                    BxWK�
  
�          @a녿�ff@P��=#�
?(��B�uÿ�ff@HQ�?fffAnffB�
=                                    BxWK��  �          @c33�5@U��\)����B�.�5@N�R?Q�A^{B��                                    BxWK�V  "          @Y���#�
@N{<#�
>�  Bʣ׿#�
@Fff?^�RAt��B�z�                                    BxWK��  T          @`�׿(�@Z�H=#�
?z�B�#׿(�@R�\?p��A{33B��                                    BxWKϢ  �          @^�R�aG�@Tz�>��@أ�Bѳ3�aG�@E?��\A��RB�                                    BxWK�H  T          @^�R��z�@H��>�33@��
B�zῴz�@;�?�A�z�B�u�                                    BxWK��  �          @`  ����@8��?�z�A�ffB�=q����@=q@Q�B�B�q                                    BxWK��  �          @b�\���@<(�?���A�=qB�uÿ��@{@
=B�
B                                    BxWL
:  
�          @]p����@{?У�A�33B��f���?���@�RB \)C�{                                    BxWL�  
(          @^�R���
@   ?�=qB {B����
?�
=@�B2�C ^�                                    BxWL'�  
(          @Y����z�@)��?�z�A�(�B�\)��z�@�@33B.Q�B�=                                    BxWL6,  "          @e��W
=@S33?�=qA���BЀ �W
=@9��?�BB�G�                                    BxWLD�  
�          @c�
�L��@U?n{ArffB��)�L��@>{?��
A�z�B�                                    BxWLSx  
�          @Z=q�E�@J�H?E�AT(�B�B��E�@7
=?�=qA�33B�(�                                    BxWLb  �          @c33?�=q@(��z����BX�?�=q@*=q��z���z�BmG�                                    BxWLp�  T          @c33?�\?�
=�Q��)G�B=\)?�\@\)�����RBZ
=                                    BxWLj  "          @g
=?�p�@1녿���ۮBxQ�?�p�@G��W
=�[\)B�\)                                    BxWL�  �          @s33?}p�@Y�������
=B���?}p�@hQ쾽p����\B��{                                    BxWL��  �          @u�?u@^�R��G����B�.?u@l(���z����
B��
                                    BxWL�\  
�          @z=q?��@HQ��z��뙚B�Q�?��@a녿�G��q�B��f                                    BxWL�  �          @x��?���@G
=�   ����B�  ?���@a녿�{��33B�p�                                    BxWLȨ  �          @|��?�p�@J=q� ������B�� ?�p�@e�������33B��                                    BxWL�N  
�          @{�?��@B�\��
=B�G�?��@_\)���H��  B��3                                    BxWL��  �          @w
=?���@HQ���H��\)B�W
?���@b�\�����Q�B�k�                                    BxWL��  
�          @w�?n{@E��33��HB�p�?n{@aG���z����B�\)                                    BxWM@  
�          @|��?�\)@`  ����ffB��?�\)@l(��L���=p�B�{                                    BxWM�  �          @y��?���@Y����p����\B�?���@j�H������B��3                                    BxWM �  �          @u?+�@[��Ǯ��Q�B��
?+�@n�R����\)B��=                                    BxWM/2  �          @Tz�\)@<�;u��\)B�k��\)@:=q?
=qA)��B�z�                                    BxWM=�  �          @>{��  @ff?\A���B��῀  ?�{@B2��B�L�                                    BxWML~  
Z          @U����@<��?L��Ad(�Bۙ����@(Q�?�ffA���B��                                    BxWM[$  T          @b�\�p��@Z=q    =L��B���p��@Q�?fffAl��B��f                                    BxWMi�  "          @Z=q�#�
@U��u�~{B���#�
@QG�?&ffA1G�B�33                                    BxWMxp  �          @L�Ϳ�@G
=���	p�B�.��@G�>�Q�@���B��                                    BxWM�  �          @R�\�=p�@K���{��ffB�\�=p�@J=q?�\A�B�B�                                    BxWM��  T          @U�^�R@G
=�W
=�i��B�B��^�R@N{<#�
>.{B�8R                                    BxWM�b  T          @L(��xQ�@7
=�
=q�!�B�p��xQ�@:=q>u@���B��                                    BxWM�  "          @W
=�˅>�G�@9��BqffC$^��˅��{@:�HBsp�C?��                                    BxWM��  
�          @XQ�����@#33BF=qC4�)���5@��B<{CE��                                    BxWM�T  T          @XQ��p��z�@p�B$�CAJ=�p����?�p�BG�CL�                                    BxWM��  	�          @`  �=q��@�B��CRxR�=q��z�?У�Aߙ�CZ}q                                    BxWM��  �          @dzῑ녿n{@#33BoC[.��녿���@p�BB�HCi�3                                    BxWM�F  T          @�p���
=@�\@P  BX
=B�33��
=?�{@k�B�G�C�                                     BxWN
�  "          @�{���?�Q�@p��BhffB�8R���?aG�@�z�B�8RC33                                    BxWN�  �          @�
=���?���@qG�Bf�B�  ���?aG�@���B���C�=                                    BxWN(8  
�          @�
=��  @G�@r�\Bg�B��Ϳ�  ?s33@�{B���C�3                                    BxWN6�  �          @����Q�@z�@hQ�BY{B� ��Q�?��
@��B���C�3                                    BxWNE�  T          @mp��8Q�?�=q@N{Bv��B��)�8Q�?.{@aG�B���C�)                                    BxWNT*  �          @hQ쾽p�@	��@8��BS�\Bǣ׾�p�?���@VffB��\BӔ{                                    BxWNb�  �          @W
==��
@{@G�B*=qB�Q�=��
?��
@5�Bg(�B��f                                    BxWNqv  �          @]p��W
=@3�
?��RB��B�\)�W
=@�@)��BI�B��q                                    BxWN�  
�          @aG���p�@0��@
�HB��B�\)��p�@@3�
BS��B�.                                    BxWN��  �          @Vff���R@,��?���B�
B�  ���R@@%BK33BĸR                                    BxWN�h  �          @mp��!G�@G�?�{A�RB��Ϳ!G�@!G�@'
=B4�RB�                                      BxWN�  �          @x�ÿO\)@Y��?�(�AхB��f�O\)@5�@#33B#��B�\                                    BxWN��  
�          @tz�E�@Y��?�ffA�=qBͅ�E�@7�@��B(�B�                                      BxWN�Z  
�          @p  ���@\��?��A�33Bǳ3���@>�R@(�BG�Bʽq                                    